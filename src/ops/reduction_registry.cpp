#include "pcr/ops/reduction_registry.h"
#include "pcr/ops/builtin_ops.h"
#include "pcr/core/types.h"
#include <unordered_map>
#include <cstring>
#ifdef PCR_HAS_OPENMP
#include <omp.h>
#endif

namespace pcr {

// ===========================================================================
// CPU Reference Implementations
// ===========================================================================
//
// These are single-threaded CPU implementations for each reduction op.
// When PCR_ENABLE_CUDA=ON, these will be replaced by GPU kernel launches.
//
// The template parameter Op must provide:
//   - State, identity(), combine(), merge(), finalize(), state_floats()
//   - pack_state<Op>() and unpack_state<Op>() specializations
//
// ===========================================================================

// ---------------------------------------------------------------------------
// InitState — initialize state buffer to identity values
// ---------------------------------------------------------------------------
template <typename Op>
Status init_state_cpu(float* state, int64_t tile_cells, void* /*stream*/) {
    typename Op::State identity = Op::identity();

#ifdef PCR_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < tile_cells; ++i) {
        pack_state<Op>(identity, state, i, tile_cells);
    }

    return Status::success();
}

// ---------------------------------------------------------------------------
// Accumulate — fold points into state buffer
// ---------------------------------------------------------------------------
//
// cell_indices[j] is the cell index for point j (relative to tile, sorted)
// values[j] is the measurement value for point j
// state is the tile state buffer (K * tile_cells floats, band-sequential)
//
template <typename Op>
Status accumulate_cpu(
    const uint32_t* cell_indices,
    const float*    values,
    float*          state,
    size_t          num_points,
    int64_t         tile_cells,
    void*           /*stream*/)
{
    // Simple approach: for each point, unpack state, combine, pack state.
    // Note: points are sorted by cell_index, so consecutive points often hit
    // the same cell. A more optimized version would process runs.

#ifdef PCR_HAS_OPENMP
    // OpenMP parallelization with critical section for state updates
    // This gives good performance when different threads work on different cells
    bool error_occurred = false;

    #pragma omp parallel for schedule(static) shared(error_occurred)
    for (size_t j = 0; j < num_points; ++j) {
        if (error_occurred) continue;  // Skip if error in another thread

        uint32_t cell = cell_indices[j];
        if (cell >= static_cast<uint32_t>(tile_cells)) {
            #pragma omp critical
            error_occurred = true;
            continue;
        }

        float val = values[j];

        // Critical section protects the read-modify-write sequence
        // This serializes updates to the same cell but allows parallel updates to different cells
        #pragma omp critical
        {
            typename Op::State acc = unpack_state<Op>(state, cell, tile_cells);
            acc = Op::combine(acc, val);
            pack_state<Op>(acc, state, cell, tile_cells);
        }
    }

    if (error_occurred) {
        return Status::error(StatusCode::InvalidArgument, "cell index out of range");
    }
#else
    // Single-threaded version
    for (size_t j = 0; j < num_points; ++j) {
        uint32_t cell = cell_indices[j];
        if (cell >= static_cast<uint32_t>(tile_cells)) {
            return Status::error(StatusCode::InvalidArgument, "cell index out of range");
        }

        float val = values[j];
        typename Op::State acc = unpack_state<Op>(state, cell, tile_cells);
        acc = Op::combine(acc, val);
        pack_state<Op>(acc, state, cell, tile_cells);
    }
#endif

    return Status::success();
}

// ---------------------------------------------------------------------------
// MergeState — merge src state into dst state element-wise
// ---------------------------------------------------------------------------
template <typename Op>
Status merge_state_cpu(
    float*       dst_state,
    const float* src_state,
    int64_t      tile_cells,
    void*        /*stream*/)
{
#ifdef PCR_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < tile_cells; ++i) {
        typename Op::State dst = unpack_state<Op>(dst_state, i, tile_cells);
        typename Op::State src = unpack_state<Op>(src_state, i, tile_cells);
        typename Op::State merged = Op::merge(dst, src);
        pack_state<Op>(merged, dst_state, i, tile_cells);
    }

    return Status::success();
}

// ---------------------------------------------------------------------------
// Finalize — convert state buffer to output values
// ---------------------------------------------------------------------------
template <typename Op>
Status finalize_cpu(
    const float* state,
    float*       output,
    int64_t      tile_cells,
    void*        /*stream*/)
{
#ifdef PCR_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < tile_cells; ++i) {
        typename Op::State s = unpack_state<Op>(state, i, tile_cells);
        output[i] = Op::finalize(s);
    }

    return Status::success();
}

// ===========================================================================
// ReductionInfo Registry
// ===========================================================================

// Helper to create ReductionInfo for a given Op
template <typename Op>
ReductionInfo make_reduction_info(ReductionType type) {
    ReductionInfo info;
    info.type         = type;
    info.state_floats = Op::state_floats();
    info.init_state   = init_state_cpu<Op>;
    info.accumulate   = accumulate_cpu<Op>;
    info.merge_state  = merge_state_cpu<Op>;
    info.finalize     = finalize_cpu<Op>;
    return info;
}

// Static registry
static std::unordered_map<ReductionType, ReductionInfo> g_registry = {
    {ReductionType::Sum,     make_reduction_info<SumOp>(ReductionType::Sum)},
    {ReductionType::Max,     make_reduction_info<MaxOp>(ReductionType::Max)},
    {ReductionType::Min,     make_reduction_info<MinOp>(ReductionType::Min)},
    {ReductionType::Count,   make_reduction_info<CountOp>(ReductionType::Count)},
    {ReductionType::Average, make_reduction_info<AverageOp>(ReductionType::Average)},

    // TODO: WeightedAverageOp and MostRecentOp require special accumulate kernels
    //       that take additional channel data (weight/timestamp). These need
    //       extended AccumulateFn signatures or separate kernel paths.

    // TODO: Median, Percentile, PriorityMerge not yet implemented
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const ReductionInfo* get_reduction(ReductionType type) {
    auto it = g_registry.find(type);
    if (it != g_registry.end()) {
        return &it->second;
    }
    return nullptr;
}

int reduction_state_floats(ReductionType type) {
    const ReductionInfo* info = get_reduction(type);
    if (info) {
        return info->state_floats;
    }

    // Default to 1 for unknown types (conservative)
    return 1;
}

} // namespace pcr
