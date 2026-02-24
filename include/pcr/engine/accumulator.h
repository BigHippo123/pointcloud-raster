#pragma once

#include "pcr/core/types.h"
#include "pcr/engine/tile_router.h"
#include <memory>

namespace pcr {

class Grid;
class MemoryPool;

// ---------------------------------------------------------------------------
// Accumulator — Applies a reduction operation to a TileBatch, accumulating
// results into a tile's state buffer on GPU.
//
// The state buffer is a float array of size (Op::state_floats * tile_cells).
// It persists across multiple accumulate() calls (multiple chunks/collections).
// ---------------------------------------------------------------------------
class Accumulator {
public:
    ~Accumulator();

    static std::unique_ptr<Accumulator> create(MemoryPool* pool = nullptr);

    /// Accumulate a batch of sorted points into the tile state buffer.
    ///
    /// `state` is a device float buffer, layout: band-sequential,
    ///   size = state_floats * tile_cells.
    /// `batch` contains sorted local_cell_indices and values for one tile.
    /// `tile_cells` = tile_width * tile_height (from GridConfig).
    ///
    /// The state buffer must be initialized (via InitStateFn) before first use.
    Status accumulate(ReductionType type,
                      const TileBatch& batch,
                      float* state,
                      int64_t tile_cells,
                      void* stream = nullptr);

    /// Templated version for custom ops (bypasses registry).
    template <typename Op>
    Status accumulate(const TileBatch& batch,
                      float* state,
                      int64_t tile_cells,
                      void* stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcr
