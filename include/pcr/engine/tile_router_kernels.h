#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include "pcr/engine/tile_router.h"
#include "pcr/engine/memory_pool.h"

namespace pcr {

#ifdef PCR_HAS_CUDA

// GPU kernel wrappers for tile router operations

/// Assign tile/cell indices to each point on GPU.
Status tile_router_assign_gpu(
    const double* d_x,
    const double* d_y,
    size_t num_points,
    const GridConfig& config,
    uint32_t* d_cell_indices,
    uint32_t* d_tile_indices,
    uint8_t*  d_valid_mask,
    void* stream);

/// Sort arrays by (tile_index, cell_index) using CUB radix sort.
/// Co-sorts value, weight, timestamp and optional glyph arrays.
Status tile_router_sort_gpu(
    uint32_t* d_cell_indices,
    uint32_t* d_tile_indices,
    uint8_t*  d_valid_mask,
    float*    d_values,
    float*    d_weights,
    float*    d_timestamps,
    size_t    num_points,
    MemoryPool* pool,
    void* stream,
    GlyphSortArrays* glyph = nullptr);

/// Convert global cell indices to local (within-tile) cell indices on GPU.
Status tile_router_global_to_local_gpu(
    uint32_t* d_cell_indices,
    const uint32_t* d_tile_indices,
    size_t num_points,
    int grid_width,
    int grid_height,
    int tile_width,
    int tile_height,
    void* stream);

#endif // PCR_HAS_CUDA

} // namespace pcr
