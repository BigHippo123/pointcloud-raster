#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include <string>
#include <memory>

namespace pcr {

class Grid;

// ---------------------------------------------------------------------------
// TileManager — Manages tile state persistence for out-of-core processing.
//
// Each tile's reduction state is stored as a binary file on disk.
// An LRU cache keeps frequently-accessed tiles in pinned host memory
// to minimize disk I/O.
//
// Workflow:
//   1. acquire(tile) → pinned host pointer to state (loaded from disk or cache)
//   2. Upload to GPU, accumulate, download back
//   3. release(tile) → marks dirty, will be flushed to disk when evicted
//   4. flush_all() → write all dirty tiles to disk (end of pipeline)
// ---------------------------------------------------------------------------

struct TileManagerConfig {
    std::string    state_dir;         // directory for tile state files
    size_t         cache_size_bytes;  // max pinned host memory for LRU cache
    int            state_floats;      // per-cell state size (from reduction op)
    GridConfig     grid_config;       // for tile dimensions
    MemoryLocation memory_location = MemoryLocation::Host;  // where to allocate tile state
    void*          cuda_stream = nullptr;  // CUDA stream for async transfers (if using Device)
};

class TileManager {
public:
    ~TileManager();

    static std::unique_ptr<TileManager> create(const TileManagerConfig& config);

    /// Acquire tile state into pinned host memory.
    /// If tile has no prior state, initializes to identity (via ReductionType).
    /// Returns host pointer to state buffer (state_floats * tile_cells floats).
    /// Tile is pinned in cache until release().
    Status acquire(TileIndex tile, ReductionType type, float** out_host_ptr);

    /// Release tile back to cache, marking it dirty (modified).
    /// The tile may be evicted to disk later if cache is full.
    Status release(TileIndex tile);

    /// Flush all dirty tiles to disk.
    Status flush_all();

    /// Discard all cached tiles (does NOT flush — call flush_all first).
    void clear_cache();

    /// Check if tile has any accumulated state (file exists or in cache).
    bool tile_has_state(TileIndex tile) const;

    /// Delete all state files. Resets the entire accumulation.
    Status reset();

    // -- Stats --------------------------------------------------------------
    size_t cache_hits()     const;
    size_t cache_misses()   const;
    size_t tiles_on_disk()  const;
    size_t tiles_in_cache() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcr
