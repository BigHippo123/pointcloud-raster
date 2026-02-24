#pragma once

#include "pcr/core/types.h"
#include <cstdint>

namespace pcr {

// ---------------------------------------------------------------------------
// Grid merge — element-wise merge of two state buffers.
//
// Used when accumulating in chunks: each chunk produces partial state,
// which is merged into the running tile state.
//
// Also used for multi-collection merge when processing collections
// independently and combining afterwards.
//
// dst_state = merge(dst_state, src_state) for each cell.
// ---------------------------------------------------------------------------

/// Merge src into dst using the specified reduction's merge function.
/// Both buffers must be on device, same size: state_floats * tile_cells.
Status merge_tile_state(ReductionType type,
                        float* dst_state,
                        const float* src_state,
                        int64_t tile_cells,
                        void* stream = nullptr);

/// Finalize state → output values for a tile.
/// `state` is device input (state_floats * tile_cells).
/// `output` is device output (tile_cells floats).
Status finalize_tile(ReductionType type,
                     const float* state,
                     float* output,
                     int64_t tile_cells,
                     void* stream = nullptr);

/// Initialize state buffer to identity values for the given reduction.
Status init_tile_state(ReductionType type,
                       float* state,
                       int64_t tile_cells,
                       void* stream = nullptr);

} // namespace pcr
