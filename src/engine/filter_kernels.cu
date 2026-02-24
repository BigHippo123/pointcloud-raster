#include "pcr/engine/filter_kernels.h"

#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#include <cub/device/device_select.cuh>

namespace pcr {

// ---------------------------------------------------------------------------
// Device-side predicate data (small, constant-size per predicate)
// ---------------------------------------------------------------------------
struct DevicePredicate {
    CompareOp op;
    float     value;
    float     value_set[16];  // max 16 set values on device
    int       value_set_size;
};

// ---------------------------------------------------------------------------
// Kernel: evaluate predicates for each point
// ---------------------------------------------------------------------------
__global__ void kernel_evaluate_predicates(
    const float* const* channel_ptrs,   // array of channel data pointers
    const DevicePredicate* predicates,   // array of predicates
    int num_predicates,
    size_t num_points,
    uint8_t* flags)                      // output: 1 = pass, 0 = reject
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    bool passes = true;
    for (int p = 0; p < num_predicates && passes; ++p) {
        float val = channel_ptrs[p][i];
        const DevicePredicate& pred = predicates[p];

        switch (pred.op) {
            case CompareOp::Equal:        passes = (val == pred.value); break;
            case CompareOp::NotEqual:     passes = (val != pred.value); break;
            case CompareOp::Less:         passes = (val < pred.value); break;
            case CompareOp::LessEqual:    passes = (val <= pred.value); break;
            case CompareOp::Greater:      passes = (val > pred.value); break;
            case CompareOp::GreaterEqual: passes = (val >= pred.value); break;
            case CompareOp::InSet: {
                bool found = false;
                for (int s = 0; s < pred.value_set_size; ++s) {
                    if (val == pred.value_set[s]) { found = true; break; }
                }
                passes = found;
                break;
            }
            case CompareOp::NotInSet: {
                bool found = false;
                for (int s = 0; s < pred.value_set_size; ++s) {
                    if (val == pred.value_set[s]) { found = true; break; }
                }
                passes = !found;
                break;
            }
        }
    }

    flags[i] = passes ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Kernel: convert flags to indices (for use with CUB select)
// ---------------------------------------------------------------------------
__global__ void kernel_iota(uint32_t* indices, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        indices[i] = static_cast<uint32_t>(i);
    }
}

// ---------------------------------------------------------------------------
// Host function: GPU filter implementation
// ---------------------------------------------------------------------------
Status filter_points_gpu(
    const float* const* d_channel_ptrs,   // device array of channel pointers
    const DeviceFilterData& filter_data,
    size_t num_points,
    uint32_t** out_indices,
    size_t* out_count,
    MemoryPool* pool,
    void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    constexpr int BLOCK = 256;
    int grid = static_cast<int>((num_points + BLOCK - 1) / BLOCK);

    // Allocate flags buffer
    uint8_t* d_flags = nullptr;
    uint32_t* d_iota = nullptr;
    uint32_t* d_selected = nullptr;
    size_t* d_num_selected = nullptr;
    bool pool_alloc = (pool != nullptr);

    if (pool_alloc) {
        d_flags = static_cast<uint8_t*>(pool->allocate(num_points));
        d_iota = static_cast<uint32_t*>(pool->allocate(num_points * sizeof(uint32_t)));
        d_selected = static_cast<uint32_t*>(pool->allocate(num_points * sizeof(uint32_t)));
        d_num_selected = static_cast<size_t*>(pool->allocate(sizeof(size_t)));
    } else {
        cudaMalloc(&d_flags, num_points);
        cudaMalloc(&d_iota, num_points * sizeof(uint32_t));
        cudaMalloc(&d_selected, num_points * sizeof(uint32_t));
        cudaMalloc(&d_num_selected, sizeof(size_t));
    }

    if (!d_flags || !d_iota || !d_selected || !d_num_selected) {
        if (!pool_alloc) {
            cudaFree(d_flags);
            cudaFree(d_iota);
            cudaFree(d_selected);
            cudaFree(d_num_selected);
        }
        return Status::error(StatusCode::OutOfMemory, "filter_points_gpu: allocation failed");
    }

    // Generate iota indices
    kernel_iota<<<grid, BLOCK, 0, s>>>(d_iota, num_points);

    // Evaluate predicates
    kernel_evaluate_predicates<<<grid, BLOCK, 0, s>>>(
        d_channel_ptrs,
        static_cast<const DevicePredicate*>(filter_data.d_predicates),
        filter_data.num_predicates,
        num_points,
        d_flags);

    // CUB stream compaction: select indices where flag == 1
    // First determine temp storage size
    size_t temp_bytes = 0;
    cub::DeviceSelect::Flagged(nullptr, temp_bytes,
        d_iota, d_flags, d_selected, d_num_selected,
        static_cast<int>(num_points), s);

    void* d_temp = nullptr;
    if (pool_alloc) {
        d_temp = pool->allocate(temp_bytes);
    } else {
        cudaMalloc(&d_temp, temp_bytes);
    }

    if (!d_temp) {
        if (!pool_alloc) {
            cudaFree(d_flags);
            cudaFree(d_iota);
            cudaFree(d_selected);
            cudaFree(d_num_selected);
        }
        return Status::error(StatusCode::OutOfMemory, "filter_points_gpu: CUB temp allocation failed");
    }

    cub::DeviceSelect::Flagged(d_temp, temp_bytes,
        d_iota, d_flags, d_selected, d_num_selected,
        static_cast<int>(num_points), s);

    // Copy count back to host
    size_t h_num_selected = 0;
    cudaMemcpyAsync(&h_num_selected, d_num_selected, sizeof(size_t),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    *out_indices = d_selected;
    *out_count = h_num_selected;

    // Free temporaries (not selected output)
    if (!pool_alloc) {
        cudaFree(d_flags);
        cudaFree(d_iota);
        cudaFree(d_temp);
        cudaFree(d_num_selected);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            std::string("filter_points_gpu: ") + cudaGetErrorString(err));
    }

    return Status::success();
}

// ---------------------------------------------------------------------------
// Upload filter data to device
// ---------------------------------------------------------------------------
Status upload_filter_data(const FilterSpec& spec,
                          const std::vector<const float*>& d_channel_ptrs,
                          DeviceFilterData& out) {
    int num_preds = static_cast<int>(spec.predicates.size());

    // Build host-side predicates
    std::vector<DevicePredicate> h_predicates(num_preds);
    for (int i = 0; i < num_preds; ++i) {
        const auto& pred = spec.predicates[i];
        h_predicates[i].op = pred.op;
        h_predicates[i].value = pred.value;
        h_predicates[i].value_set_size = std::min(static_cast<int>(pred.value_set.size()), 16);
        for (int j = 0; j < h_predicates[i].value_set_size; ++j) {
            h_predicates[i].value_set[j] = pred.value_set[j];
        }
    }

    // Upload predicates
    DevicePredicate* d_preds = nullptr;
    cudaMalloc(&d_preds, num_preds * sizeof(DevicePredicate));
    cudaMemcpy(d_preds, h_predicates.data(), num_preds * sizeof(DevicePredicate),
               cudaMemcpyHostToDevice);

    // Upload channel pointer array
    const float** d_ch_ptrs = nullptr;
    cudaMalloc(&d_ch_ptrs, num_preds * sizeof(const float*));
    cudaMemcpy(d_ch_ptrs, d_channel_ptrs.data(), num_preds * sizeof(const float*),
               cudaMemcpyHostToDevice);

    out.d_predicates = d_preds;
    out.d_channel_ptrs = d_ch_ptrs;
    out.num_predicates = num_preds;

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            std::string("upload_filter_data: ") + cudaGetErrorString(err));
    }

    return Status::success();
}

void free_filter_data(DeviceFilterData& data) {
    if (data.d_predicates) cudaFree(data.d_predicates);
    if (data.d_channel_ptrs) cudaFree(const_cast<float**>(data.d_channel_ptrs));
    data.d_predicates = nullptr;
    data.d_channel_ptrs = nullptr;
    data.num_predicates = 0;
}

} // namespace pcr

#endif // PCR_HAS_CUDA
