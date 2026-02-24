#pragma once

#include "pcr/core/types.h"
#include <functional>

namespace pcr {

// Forward declarations
class PointCloud;
class Grid;
struct GridConfig;

// ---------------------------------------------------------------------------
// ReductionRegistry — maps ReductionType enum to templated kernel launches.
//
// This is the bridge between the Python-facing enum API and the
// compile-time templated CUDA kernels. Each registered reduction
// provides function pointers for accumulate, merge, and finalize.
// ---------------------------------------------------------------------------

/// Function signatures for type-erased reduction operations.
/// These are called by the pipeline; implementations live in accumulator.cu / grid_merge.cu.

/// Accumulate points into tile state buffer on GPU.
///   cell_indices: sorted cell indices (relative to tile), length = num_points
///   values:       sorted values corresponding to cell_indices
///   state:        tile state buffer (K * tile_cells floats), device memory
///   num_points:   number of points in this chunk for this tile
///   tile_cells:   total cells in this tile
///   stream:       CUDA stream
using AccumulateFn = std::function<Status(
    const uint32_t* cell_indices,
    const float*    values,
    float*          state,
    size_t          num_points,
    int64_t         tile_cells,
    void*           stream)>;

/// Merge two state buffers element-wise: dst = merge(dst, src).
using MergeStateFn = std::function<Status(
    float*       dst_state,
    const float* src_state,
    int64_t      tile_cells,
    void*        stream)>;

/// Finalize state → output values: output[i] = finalize(state[i]).
using FinalizeFn = std::function<Status(
    const float* state,
    float*       output,
    int64_t      tile_cells,
    void*        stream)>;

/// Initialize state buffer to identity values.
using InitStateFn = std::function<Status(
    float*  state,
    int64_t tile_cells,
    void*   stream)>;

// ---------------------------------------------------------------------------
// ReductionInfo — everything needed to execute a reduction at runtime
// ---------------------------------------------------------------------------
struct ReductionInfo {
    ReductionType type;
    int           state_floats;     // number of float fields per cell in state
    AccumulateFn  accumulate;
    MergeStateFn  merge_state;
    FinalizeFn    finalize;
    InitStateFn   init_state;
};

// ---------------------------------------------------------------------------
// Registry access
// ---------------------------------------------------------------------------

/// Get reduction info for a builtin type. Returns nullptr for Custom.
const ReductionInfo* get_reduction(ReductionType type);

/// Number of state floats required for a given reduction type.
int reduction_state_floats(ReductionType type);

} // namespace pcr
