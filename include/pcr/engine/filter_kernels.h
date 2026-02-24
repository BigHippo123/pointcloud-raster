#pragma once

#include "pcr/core/types.h"
#include "pcr/engine/filter.h"
#include "pcr/engine/memory_pool.h"
#include <vector>

namespace pcr {

#ifdef PCR_HAS_CUDA

struct DeviceFilterData {
    void*        d_predicates   = nullptr;  // DevicePredicate array on device
    const float** d_channel_ptrs = nullptr;  // channel pointer array on device
    int          num_predicates = 0;
};

Status filter_points_gpu(
    const float* const* d_channel_ptrs,
    const DeviceFilterData& filter_data,
    size_t num_points,
    uint32_t** out_indices,
    size_t* out_count,
    MemoryPool* pool,
    void* stream);

Status upload_filter_data(const FilterSpec& spec,
                          const std::vector<const float*>& d_channel_ptrs,
                          DeviceFilterData& out);

void free_filter_data(DeviceFilterData& data);

#endif // PCR_HAS_CUDA

} // namespace pcr
