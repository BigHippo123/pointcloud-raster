#include "pcr/engine/grid_merge.h"
#include "pcr/ops/builtin_ops.h"
#include "pcr/core/types.h"
#include "pcr/ops/reduction_registry.h"

#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>

namespace pcr {

// ---------------------------------------------------------------------------
// GPU Kernels for Grid Merge Operations
// ---------------------------------------------------------------------------

/// Initialize state buffer to identity values
template <typename Op>
__global__ void init_state_kernel(float* state, int64_t tile_cells) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < tile_cells) {
        typename Op::State identity = Op::identity();
        pack_state<Op>(identity, state, i, tile_cells);
    }
}

/// Merge src state into dst state element-wise
template <typename Op>
__global__ void merge_state_kernel(float* dst_state, const float* src_state, int64_t tile_cells) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < tile_cells) {
        typename Op::State dst = unpack_state<Op>(dst_state, i, tile_cells);
        typename Op::State src = unpack_state<Op>(src_state, i, tile_cells);
        typename Op::State merged = Op::merge(dst, src);
        pack_state<Op>(merged, dst_state, i, tile_cells);
    }
}

/// Finalize state → output values
template <typename Op>
__global__ void finalize_kernel(const float* state, float* output, int64_t tile_cells) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < tile_cells) {
        typename Op::State s = unpack_state<Op>(state, i, tile_cells);
        output[i] = Op::finalize(s);
    }
}

// ---------------------------------------------------------------------------
// Kernel Launchers (template wrappers)
// ---------------------------------------------------------------------------

template <typename Op>
Status init_state_gpu_impl(float* state, int64_t tile_cells, void* stream) {
    if (tile_cells == 0) return Status::success();

    constexpr int block_size = 256;
    int64_t num_blocks = (tile_cells + block_size - 1) / block_size;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    init_state_kernel<Op><<<num_blocks, block_size, 0, cuda_stream>>>(state, tile_cells);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            "init_state_kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return Status::success();
}

template <typename Op>
Status merge_state_gpu_impl(float* dst_state, const float* src_state,
                             int64_t tile_cells, void* stream) {
    if (tile_cells == 0) return Status::success();

    constexpr int block_size = 256;
    int64_t num_blocks = (tile_cells + block_size - 1) / block_size;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    merge_state_kernel<Op><<<num_blocks, block_size, 0, cuda_stream>>>(
        dst_state, src_state, tile_cells);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            "merge_state_kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return Status::success();
}

template <typename Op>
Status finalize_gpu_impl(const float* state, float* output,
                         int64_t tile_cells, void* stream) {
    if (tile_cells == 0) return Status::success();

    constexpr int block_size = 256;
    int64_t num_blocks = (tile_cells + block_size - 1) / block_size;

    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    finalize_kernel<Op><<<num_blocks, block_size, 0, cuda_stream>>>(
        state, output, tile_cells);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            "finalize_kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    return Status::success();
}

// ---------------------------------------------------------------------------
// Public API: Dispatchers based on ReductionType
// ---------------------------------------------------------------------------

Status init_tile_state(ReductionType type, float* state,
                       int64_t tile_cells, void* stream) {
    switch (type) {
        case ReductionType::Sum:
            return init_state_gpu_impl<SumOp>(state, tile_cells, stream);
        case ReductionType::Max:
            return init_state_gpu_impl<MaxOp>(state, tile_cells, stream);
        case ReductionType::Min:
            return init_state_gpu_impl<MinOp>(state, tile_cells, stream);
        case ReductionType::Count:
            return init_state_gpu_impl<CountOp>(state, tile_cells, stream);
        case ReductionType::Average:
            return init_state_gpu_impl<AverageOp>(state, tile_cells, stream);
        case ReductionType::WeightedAverage:
            return init_state_gpu_impl<WeightedAverageOp>(state, tile_cells, stream);
        case ReductionType::MostRecent:
            return init_state_gpu_impl<MostRecentOp>(state, tile_cells, stream);
        default:
            return Status::error(StatusCode::NotImplemented,
                "init_tile_state: unsupported reduction type");
    }
}

Status merge_tile_state(ReductionType type, float* dst_state,
                        const float* src_state, int64_t tile_cells, void* stream) {
    switch (type) {
        case ReductionType::Sum:
            return merge_state_gpu_impl<SumOp>(dst_state, src_state, tile_cells, stream);
        case ReductionType::Max:
            return merge_state_gpu_impl<MaxOp>(dst_state, src_state, tile_cells, stream);
        case ReductionType::Min:
            return merge_state_gpu_impl<MinOp>(dst_state, src_state, tile_cells, stream);
        case ReductionType::Count:
            return merge_state_gpu_impl<CountOp>(dst_state, src_state, tile_cells, stream);
        case ReductionType::Average:
            return merge_state_gpu_impl<AverageOp>(dst_state, src_state, tile_cells, stream);
        case ReductionType::WeightedAverage:
            return merge_state_gpu_impl<WeightedAverageOp>(dst_state, src_state, tile_cells, stream);
        case ReductionType::MostRecent:
            return merge_state_gpu_impl<MostRecentOp>(dst_state, src_state, tile_cells, stream);
        default:
            return Status::error(StatusCode::NotImplemented,
                "merge_tile_state: unsupported reduction type");
    }
}

Status finalize_tile(ReductionType type, const float* state,
                     float* output, int64_t tile_cells, void* stream) {
    switch (type) {
        case ReductionType::Sum:
            return finalize_gpu_impl<SumOp>(state, output, tile_cells, stream);
        case ReductionType::Max:
            return finalize_gpu_impl<MaxOp>(state, output, tile_cells, stream);
        case ReductionType::Min:
            return finalize_gpu_impl<MinOp>(state, output, tile_cells, stream);
        case ReductionType::Count:
            return finalize_gpu_impl<CountOp>(state, output, tile_cells, stream);
        case ReductionType::Average:
            return finalize_gpu_impl<AverageOp>(state, output, tile_cells, stream);
        case ReductionType::WeightedAverage:
            return finalize_gpu_impl<WeightedAverageOp>(state, output, tile_cells, stream);
        case ReductionType::MostRecent:
            return finalize_gpu_impl<MostRecentOp>(state, output, tile_cells, stream);
        default:
            return Status::error(StatusCode::NotImplemented,
                "finalize_tile: unsupported reduction type");
    }
}

} // namespace pcr

#else

// ---------------------------------------------------------------------------
// CPU-only build: stubs that return NotImplemented
// ---------------------------------------------------------------------------

namespace pcr {

Status init_tile_state(ReductionType, float*, int64_t, void*) {
    return Status::error(StatusCode::NotImplemented,
        "init_tile_state: CUDA not enabled");
}

Status merge_tile_state(ReductionType, float*, const float*, int64_t, void*) {
    return Status::error(StatusCode::NotImplemented,
        "merge_tile_state: CUDA not enabled");
}

Status finalize_tile(ReductionType, const float*, float*, int64_t, void*) {
    return Status::error(StatusCode::NotImplemented,
        "finalize_tile: CUDA not enabled");
}

} // namespace pcr

#endif // PCR_HAS_CUDA
