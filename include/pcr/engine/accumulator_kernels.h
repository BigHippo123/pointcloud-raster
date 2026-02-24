#pragma once

#include "pcr/core/types.h"
#include "pcr/engine/tile_router.h"
#include "pcr/engine/memory_pool.h"

namespace pcr {

#ifdef PCR_HAS_CUDA

/// GPU accumulate: fold sorted points into tile state buffer.
/// Points must be sorted by local_cell_index within the batch.
Status accumulate_gpu(
    ReductionType type,
    const uint32_t* d_cell_indices,
    const float*    d_values,
    const float*    d_weights,
    const float*    d_timestamps,
    size_t          num_points,
    float*          d_state,
    int64_t         tile_cells,
    MemoryPool*     pool,
    void*           stream);

#endif // PCR_HAS_CUDA

} // namespace pcr
