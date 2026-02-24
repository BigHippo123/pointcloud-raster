#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include <vector>
#include <memory>

namespace pcr {

class PointCloud;
class MemoryPool;

// ---------------------------------------------------------------------------
// TileAssignment — result of routing points to tiles.
//
// For a chunk of N points, produces:
//   - cell_indices[N]:  the flattened cell index within the FULL grid
//   - tile_indices[N]:  which tile each point belongs to
//   - valid_mask[N]:    false if point is outside grid bounds
//
// After routing, points are sorted by tile and then by cell within tile,
// so the accumulator can process one tile at a time.
// ---------------------------------------------------------------------------
struct TileAssignment {
    uint32_t* cell_indices  = nullptr;   // length = num_points
    uint32_t* tile_indices  = nullptr;   // packed as (tile_row * tiles_x + tile_col)
    uint8_t*  valid_mask    = nullptr;   // 1 = inside grid
    size_t    num_points    = 0;
    MemoryLocation location = MemoryLocation::Host;  // memory location of pointers
};

// ---------------------------------------------------------------------------
// TileBatch — sorted segment of points for one tile.
// Points within the batch are sorted by local cell index.
// ---------------------------------------------------------------------------
struct TileBatch {
    TileIndex tile;
    uint32_t* local_cell_indices = nullptr;  // cell index relative to tile origin
    float*    values             = nullptr;  // the value channel to reduce
    float*    weights            = nullptr;  // optional weight channel (nullptr if unused)
    float*    timestamps         = nullptr;  // optional timestamp channel (nullptr if unused)
    size_t    num_points         = 0;
    MemoryLocation location      = MemoryLocation::Host;  // memory location of pointers
};

// ---------------------------------------------------------------------------
// TileRouter — assigns points to tiles and produces sorted per-tile batches.
// ---------------------------------------------------------------------------
class TileRouter {
public:
    ~TileRouter();

    static std::unique_ptr<TileRouter> create(const GridConfig& config,
                                              MemoryPool* pool = nullptr);

    /// Phase 1: Compute cell and tile indices for all points.
    /// Points must be on device. Coordinates are read from cloud.x(), cloud.y().
    Status assign(const PointCloud& cloud,
                  TileAssignment& out,
                  void* stream = nullptr);

    /// Phase 2: Sort points by tile, then by cell within tile.
    /// Reorders cell_indices and co-sorts value/weight/timestamp arrays.
    /// After this call, you can extract TileBatches.
    Status sort(TileAssignment& assignment,
                float* values,
                float* weights,          // nullptr if not needed
                float* timestamps,       // nullptr if not needed
                void* stream = nullptr);

    /// Phase 3: Extract per-tile batches from the sorted arrays.
    /// Returns one TileBatch per tile that has points.
    /// Batches reference the sorted arrays (no copy).
    Status extract_batches(const TileAssignment& assignment,
                           float* sorted_values,
                           float* sorted_weights,
                           float* sorted_timestamps,
                           std::vector<TileBatch>& out_batches);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcr
