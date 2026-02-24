#pragma once

#include "pcr/core/types.h"
#include <cstddef>

namespace pcr {

// ---------------------------------------------------------------------------
// GridConfig — Defines the 2D output grid: bounds, resolution, CRS, tiling.
//
// The affine transform maps pixel (col, row) to world (x, y):
//   world_x = origin_x + col * cell_size_x
//   world_y = origin_y + row * cell_size_y   (cell_size_y typically negative)
//
// Tiling subdivides the grid into rectangular tiles for out-of-core processing.
// ---------------------------------------------------------------------------
struct GridConfig {
    // -- Spatial extent -----------------------------------------------------
    BBox   bounds;                     // world coordinate bounding box
    CRS    crs;                        // coordinate reference system

    // -- Resolution ---------------------------------------------------------
    double cell_size_x = 1.0;          // meters (or CRS units) per cell, X
    double cell_size_y = -1.0;         // typically negative (north-up convention)

    // -- Computed dimensions (call compute_dimensions() after setting above) -
    int    width  = 0;                 // number of columns
    int    height = 0;                 // number of rows

    // -- Nodata -------------------------------------------------------------
    NoDataPolicy nodata;

    // -- Tiling (for out-of-core processing) --------------------------------
    int    tile_width  = 4096;         // cells per tile, X
    int    tile_height = 4096;         // cells per tile, Y
    int    tiles_x     = 0;            // number of tile columns (computed)
    int    tiles_y     = 0;            // number of tile rows (computed)

    // -- Methods ------------------------------------------------------------

    /// Compute width, height, tiles_x, tiles_y from bounds and cell_size.
    /// Must be called after setting bounds and cell_size.
    void compute_dimensions();

    /// World coord → grid cell. Returns false if outside grid.
    bool world_to_cell(double wx, double wy, int& col, int& row) const;

    /// Grid cell → world coord (cell center).
    void cell_to_world(int col, int row, double& wx, double& wy) const;

    /// Grid cell → tile index.
    TileIndex cell_to_tile(int col, int row) const;

    /// Bounding box of a specific tile in world coordinates.
    BBox tile_bounds(TileIndex idx) const;

    /// Cell range within the full grid for a specific tile.
    /// Returns (col_start, row_start, col_count, row_count).
    void tile_cell_range(TileIndex idx,
                         int& col_start, int& row_start,
                         int& col_count, int& row_count) const;

    /// Total number of tiles.
    int total_tiles() const { return tiles_x * tiles_y; }

    /// Total number of cells.
    int64_t total_cells() const { return static_cast<int64_t>(width) * height; }

    /// 6-element GDAL-style geotransform:
    /// [origin_x, cell_size_x, 0, origin_y + height*|cell_size_y|, 0, cell_size_y]
    void gdal_geotransform(double gt[6]) const;

    /// Validate configuration consistency.
    Status validate() const;
};

} // namespace pcr
