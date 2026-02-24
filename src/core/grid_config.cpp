#include "pcr/core/grid_config.h"
#include <cmath>
#include <algorithm>

namespace pcr {

void GridConfig::compute_dimensions() {
    // Compute grid dimensions from bounds and cell size
    if (!bounds.valid()) {
        width = height = 0;
        tiles_x = tiles_y = 0;
        return;
    }

    // Width and height in cells
    width  = static_cast<int>(std::ceil(bounds.width()  / std::abs(cell_size_x)));
    height = static_cast<int>(std::ceil(bounds.height() / std::abs(cell_size_y)));

    // Number of tiles needed to cover the grid
    tiles_x = (width  + tile_width  - 1) / tile_width;
    tiles_y = (height + tile_height - 1) / tile_height;
}

bool GridConfig::world_to_cell(double wx, double wy, int& col, int& row) const {
    // Check bounds first
    if (!bounds.contains(wx, wy)) {
        return false;
    }

    // Transform world coords to grid cell indices
    // origin is at bounds.min_x, bounds.max_y (top-left, since cell_size_y is negative)
    double origin_x = bounds.min_x;
    double origin_y = bounds.max_y;  // top edge

    col = static_cast<int>(std::floor((wx - origin_x) / cell_size_x));
    row = static_cast<int>(std::floor((wy - origin_y) / cell_size_y));

    // Clamp to valid range (handles floating point edge cases)
    col = std::max(0, std::min(col, width - 1));
    row = std::max(0, std::min(row, height - 1));

    return true;
}

void GridConfig::cell_to_world(int col, int row, double& wx, double& wy) const {
    // Convert grid cell to world coords (cell center)
    double origin_x = bounds.min_x;
    double origin_y = bounds.max_y;

    wx = origin_x + (col + 0.5) * cell_size_x;
    wy = origin_y + (row + 0.5) * cell_size_y;
}

TileIndex GridConfig::cell_to_tile(int col, int row) const {
    TileIndex idx;
    idx.col = col / tile_width;
    idx.row = row / tile_height;
    return idx;
}

BBox GridConfig::tile_bounds(TileIndex idx) const {
    // Get cell range for this tile
    int col_start, row_start, col_count, row_count;
    tile_cell_range(idx, col_start, row_start, col_count, row_count);

    // Convert corner cells to world coords
    double origin_x = bounds.min_x;
    double origin_y = bounds.max_y;

    BBox bbox;
    bbox.min_x = origin_x + col_start * cell_size_x;
    bbox.max_x = origin_x + (col_start + col_count) * cell_size_x;

    // cell_size_y is negative, so top/bottom are swapped
    bbox.max_y = origin_y + row_start * cell_size_y;
    bbox.min_y = origin_y + (row_start + row_count) * cell_size_y;

    return bbox;
}

void GridConfig::tile_cell_range(TileIndex idx,
                                  int& col_start, int& row_start,
                                  int& col_count, int& row_count) const {
    // Starting cell indices
    col_start = idx.col * tile_width;
    row_start = idx.row * tile_height;

    // Tile dimensions (clamped to grid bounds)
    col_count = std::min(tile_width,  width  - col_start);
    row_count = std::min(tile_height, height - row_start);
}

void GridConfig::gdal_geotransform(double gt[6]) const {
    // GDAL geotransform format:
    // gt[0] = origin_x (top-left corner X)
    // gt[1] = cell_size_x (pixel width)
    // gt[2] = 0 (row rotation, typically 0)
    // gt[3] = origin_y (top-left corner Y)
    // gt[4] = 0 (column rotation, typically 0)
    // gt[5] = cell_size_y (pixel height, negative for north-up)

    double origin_x = bounds.min_x;
    double origin_y = bounds.max_y;  // top edge (north-up convention)

    gt[0] = origin_x;
    gt[1] = cell_size_x;
    gt[2] = 0.0;
    gt[3] = origin_y;
    gt[4] = 0.0;
    gt[5] = cell_size_y;
}

Status GridConfig::validate() const {
    // Check bounds
    if (!bounds.valid()) {
        return Status::error(StatusCode::InvalidArgument,
                           "Invalid bounds: max < min");
    }

    // Check cell sizes are non-zero
    if (cell_size_x == 0.0 || cell_size_y == 0.0) {
        return Status::error(StatusCode::InvalidArgument,
                           "Cell size cannot be zero");
    }

    // Check tile dimensions are positive
    if (tile_width <= 0 || tile_height <= 0) {
        return Status::error(StatusCode::InvalidArgument,
                           "Tile dimensions must be positive");
    }

    // Check grid dimensions are computed
    if (width <= 0 || height <= 0) {
        return Status::error(StatusCode::InvalidArgument,
                           "Grid dimensions not computed or invalid. Call compute_dimensions()");
    }

    // Check CRS is valid
    if (!crs.is_valid()) {
        return Status::error(StatusCode::CrsError,
                           "CRS is not valid");
    }

    return Status::success();
}

} // namespace pcr
