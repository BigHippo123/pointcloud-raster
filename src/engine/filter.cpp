#include "pcr/engine/filter.h"
#include "pcr/core/point_cloud.h"
#include "pcr/engine/memory_pool.h"
#include <cstring>
#include <algorithm>

#ifdef PCR_HAS_CUDA
#include "pcr/engine/filter_kernels.h"
#include <cuda_runtime.h>
#endif

#ifdef PCR_HAS_OPENMP
#include <omp.h>
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// FilterSpec convenience methods
// ---------------------------------------------------------------------------
FilterSpec& FilterSpec::add(const std::string& channel, CompareOp op, float value) {
    predicates.push_back({channel, op, value, {}});
    return *this;
}

FilterSpec& FilterSpec::add_in_set(const std::string& channel, const std::vector<float>& values) {
    predicates.push_back({channel, CompareOp::InSet, 0.0f, values});
    return *this;
}

// ---------------------------------------------------------------------------
// Predicate evaluation helpers
// ---------------------------------------------------------------------------
static bool evaluate_predicate(const FilterPredicate& pred, float value) {
    switch (pred.op) {
        case CompareOp::Equal:
            return value == pred.value;
        case CompareOp::NotEqual:
            return value != pred.value;
        case CompareOp::Less:
            return value < pred.value;
        case CompareOp::LessEqual:
            return value <= pred.value;
        case CompareOp::Greater:
            return value > pred.value;
        case CompareOp::GreaterEqual:
            return value >= pred.value;
        case CompareOp::InSet:
            return std::find(pred.value_set.begin(), pred.value_set.end(), value) != pred.value_set.end();
        case CompareOp::NotInSet:
            return std::find(pred.value_set.begin(), pred.value_set.end(), value) == pred.value_set.end();
    }
    return false;
}

// ---------------------------------------------------------------------------
// CPU implementation (always available)
// ---------------------------------------------------------------------------
static Status filter_points_cpu(const PointCloud& cloud,
                                 const FilterSpec& spec,
                                 uint32_t** out_indices,
                                 size_t*    out_count,
                                 MemoryPool* pool)
{
    // Validate input
    if (!out_indices || !out_count) {
        return Status::error(StatusCode::InvalidArgument,
            "filter_points: out_indices and out_count must not be null");
    }

    // Check cloud memory location
    if (cloud.location() != MemoryLocation::Host) {
        return Status::error(StatusCode::InvalidArgument,
            "CPU filter requires Host memory location");
    }

    size_t num_points = cloud.count();

    // If no predicates, all points pass
    if (spec.empty()) {
        uint32_t* indices = static_cast<uint32_t*>(malloc(num_points * sizeof(uint32_t)));
        if (!indices) {
            return Status::error(StatusCode::InvalidArgument,
                "filter_points: failed to allocate output indices");
        }

#ifdef PCR_HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < num_points; ++i) {
            indices[i] = static_cast<uint32_t>(i);
        }

        *out_indices = indices;
        *out_count = num_points;
        return Status::success();
    }

    // Get channel pointers for all predicates
    std::vector<const float*> channel_ptrs;
    channel_ptrs.reserve(spec.predicates.size());

    for (const auto& pred : spec.predicates) {
        const void* channel_data = cloud.channel_data(pred.channel_name);
        if (!channel_data) {
            return Status::error(StatusCode::InvalidArgument,
                "filter_points: channel not found: " + pred.channel_name);
        }

        const ChannelDesc* desc = cloud.channel(pred.channel_name);
        if (!desc) {
            return Status::error(StatusCode::InvalidArgument,
                "filter_points: failed to get channel descriptor");
        }

        if (desc->dtype != DataType::Float32) {
            return Status::error(StatusCode::InvalidArgument,
                "filter_points: only Float32 channels supported for filtering");
        }

        channel_ptrs.push_back(static_cast<const float*>(channel_data));
    }

    // Temporary buffer for passing indices
    std::vector<uint32_t> temp_indices;

#ifdef PCR_HAS_OPENMP
    // Parallel reduction with thread-local buffers
    int num_threads = omp_get_max_threads();
    std::vector<std::vector<uint32_t>> thread_results(num_threads);

    // Pre-allocate thread-local buffers
    for (auto& local : thread_results) {
        local.reserve(num_points / num_threads + 1);
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_indices = thread_results[tid];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < num_points; ++i) {
            bool passes = true;

            for (size_t p = 0; p < spec.predicates.size(); ++p) {
                float value = channel_ptrs[p][i];
                if (!evaluate_predicate(spec.predicates[p], value)) {
                    passes = false;
                    break;
                }
            }

            if (passes) {
                local_indices.push_back(static_cast<uint32_t>(i));
            }
        }
    }

    // Merge thread-local results
    size_t total_size = 0;
    for (const auto& local : thread_results) {
        total_size += local.size();
    }
    temp_indices.reserve(total_size);

    for (const auto& local : thread_results) {
        temp_indices.insert(temp_indices.end(), local.begin(), local.end());
    }
#else
    // Single-threaded version
    temp_indices.reserve(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        bool passes = true;

        for (size_t p = 0; p < spec.predicates.size(); ++p) {
            float value = channel_ptrs[p][i];
            if (!evaluate_predicate(spec.predicates[p], value)) {
                passes = false;
                break;
            }
        }

        if (passes) {
            temp_indices.push_back(static_cast<uint32_t>(i));
        }
    }
#endif

    // Allocate output
    size_t output_size = temp_indices.size() * sizeof(uint32_t);
    uint32_t* indices = static_cast<uint32_t*>(malloc(output_size));
    if (!indices) {
        return Status::error(StatusCode::InvalidArgument,
            "filter_points: failed to allocate output indices");
    }

    // Copy results
    memcpy(indices, temp_indices.data(), output_size);

    *out_indices = indices;
    *out_count = temp_indices.size();

    return Status::success();
}

#ifdef PCR_HAS_CUDA

// ---------------------------------------------------------------------------
// GPU implementation
// ---------------------------------------------------------------------------
static Status filter_points_gpu_dispatch(const PointCloud& cloud,
                                          const FilterSpec& spec,
                                          uint32_t** out_indices,
                                          size_t*    out_count,
                                          MemoryPool* pool,
                                          void*      stream)
{
    // Validate input
    if (!out_indices || !out_count) {
        return Status::error(StatusCode::InvalidArgument,
            "filter_points: out_indices and out_count must not be null");
    }

    // Check cloud memory location
    if (cloud.location() != MemoryLocation::Device &&
        cloud.location() != MemoryLocation::HostPinned) {
        return Status::error(StatusCode::InvalidArgument,
            "GPU filter requires Device or HostPinned memory location");
    }

    size_t num_points = cloud.count();

    // If no predicates, all points pass - create iota on host and copy
    if (spec.empty()) {
        if (num_points == 0) {
            *out_indices = nullptr;
            *out_count = 0;
            return Status::success();
        }

        size_t bytes = num_points * sizeof(uint32_t);

        // Allocate device memory
        uint32_t* d_indices = nullptr;
        if (pool) {
            d_indices = static_cast<uint32_t*>(pool->allocate(bytes));
        } else {
            cudaMalloc(&d_indices, bytes);
        }
        if (!d_indices) {
            return Status::error(StatusCode::OutOfMemory,
                "filter_points: failed to allocate output indices");
        }

        // Create iota on host
        std::vector<uint32_t> h_indices(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            h_indices[i] = static_cast<uint32_t>(i);
        }

        // Copy to device
        cudaMemcpyAsync(d_indices, h_indices.data(), bytes,
                        cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));

        *out_indices = d_indices;
        *out_count = num_points;
        return Status::success();
    }

    // Get device channel pointers for all predicates
    std::vector<const float*> d_channel_ptrs;
    d_channel_ptrs.reserve(spec.predicates.size());

    for (const auto& pred : spec.predicates) {
        const void* channel_data = cloud.channel_data(pred.channel_name);
        if (!channel_data) {
            return Status::error(StatusCode::InvalidArgument,
                "filter_points: channel not found: " + pred.channel_name);
        }

        // Get channel type
        const ChannelDesc* desc = cloud.channel(pred.channel_name);
        if (!desc) {
            return Status::error(StatusCode::InvalidArgument,
                "filter_points: failed to get channel descriptor");
        }

        // For now, only support float channels
        if (desc->dtype != DataType::Float32) {
            return Status::error(StatusCode::InvalidArgument,
                "filter_points: only Float32 channels supported for filtering");
        }

        d_channel_ptrs.push_back(static_cast<const float*>(channel_data));
    }

    // Upload filter data to device
    DeviceFilterData filter_data;
    Status s = upload_filter_data(spec, d_channel_ptrs, filter_data);
    if (!s.ok()) return s;

    // Call GPU kernel
    s = filter_points_gpu(
        d_channel_ptrs.data(),
        filter_data,
        num_points,
        out_indices,
        out_count,
        pool,
        stream);

    // Cleanup filter data
    free_filter_data(filter_data);

    return s;
}

// Dispatcher: choose between CPU and GPU based on memory location
Status filter_points(const PointCloud& cloud,
                     const FilterSpec& spec,
                     uint32_t** out_indices,
                     size_t*    out_count,
                     MemoryPool* pool,
                     void*      stream)
{
    // Dispatch based on memory location
    if (cloud.location() == MemoryLocation::Host) {
        return filter_points_cpu(cloud, spec, out_indices, out_count, pool);
    } else {
        return filter_points_gpu_dispatch(cloud, spec, out_indices, out_count, pool, stream);
    }
}

#else

// CPU-only build: just use the CPU implementation
Status filter_points(const PointCloud& cloud,
                     const FilterSpec& spec,
                     uint32_t** out_indices,
                     size_t*    out_count,
                     MemoryPool* pool,
                     void*      /*stream*/)
{
    return filter_points_cpu(cloud, spec, out_indices, out_count, pool);
}

#endif // PCR_HAS_CUDA

} // namespace pcr
