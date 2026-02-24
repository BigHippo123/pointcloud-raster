#include "pcr/core/grid_config.h"
#include "test_helpers.h"
#include <gtest/gtest.h>

using namespace pcr;
using namespace pcr::test;

// ===========================================================================
// Dimension Computation Tests
// ===========================================================================

TEST(GridConfigTest, ComputeDimensions) {
    GridConfig cfg;
    cfg.bounds.min_x = 0.0;
    cfg.bounds.min_y = 0.0;
    cfg.bounds.max_x = 100.0;
    cfg.bounds.max_y = 100.0;
    cfg.cell_size_x = 1.0;
    cfg.cell_size_y = -1.0;
    cfg.tile_width = 32;
    cfg.tile_height = 32;

    cfg.compute_dimensions();

    EXPECT_EQ(cfg.width, 100);
    EXPECT_EQ(cfg.height, 100);
    EXPECT_EQ(cfg.tiles_x, 4);  // ceil(100/32) = 4
    EXPECT_EQ(cfg.tiles_y, 4);
}

TEST(GridConfigTest, ComputeDimensionsNonIntegerCells) {
    GridConfig cfg;
    cfg.bounds.min_x = 0.0;
    cfg.bounds.min_y = 0.0;
    cfg.bounds.max_x = 100.5;
    cfg.bounds.max_y = 100.5;
    cfg.cell_size_x = 1.0;
    cfg.cell_size_y = -1.0;

    cfg.compute_dimensions();

    EXPECT_EQ(cfg.width, 101);   // ceil(100.5) = 101
    EXPECT_EQ(cfg.height, 101);
}

TEST(GridConfigTest, ComputeDimensionsLargeCellSize) {
    GridConfig cfg;
    cfg.bounds.min_x = 0.0;
    cfg.bounds.min_y = 0.0;
    cfg.bounds.max_x = 1000.0;
    cfg.bounds.max_y = 1000.0;
    cfg.cell_size_x = 10.0;
    cfg.cell_size_y = -10.0;
    cfg.tile_width = 50;
    cfg.tile_height = 50;

    cfg.compute_dimensions();

    EXPECT_EQ(cfg.width, 100);
    EXPECT_EQ(cfg.height, 100);
    EXPECT_EQ(cfg.tiles_x, 2);  // ceil(100/50) = 2
    EXPECT_EQ(cfg.tiles_y, 2);
}

TEST(GridConfigTest, ComputeDimensionsInvalidBounds) {
    GridConfig cfg;
    // Don't set bounds (invalid)
    cfg.cell_size_x = 1.0;
    cfg.cell_size_y = -1.0;

    cfg.compute_dimensions();

    EXPECT_EQ(cfg.width, 0);
    EXPECT_EQ(cfg.height, 0);
}

// ===========================================================================
// Coordinate Transform Tests
// ===========================================================================

TEST(GridConfigTest, WorldToCellValid) {
    auto cfg = make_test_grid_config(0.0, 0.0, 100.0, 100.0, 1.0);

    int col, row;
    bool result = cfg.world_to_cell(50.0, 50.0, col, row);

    EXPECT_TRUE(result);
    EXPECT_EQ(col, 50);
    EXPECT_EQ(row, 50);
}

TEST(GridConfigTest, WorldToCellOrigin) {
    auto cfg = make_test_grid_config(0.0, 0.0, 100.0, 100.0, 1.0);

    int col, row;
    bool result = cfg.world_to_cell(0.0, 100.0, col, row);  // top-left

    EXPECT_TRUE(result);
    EXPECT_EQ(col, 0);
    EXPECT_EQ(row, 0);
}

TEST(GridConfigTest, WorldToCellOutsideBounds) {
    auto cfg = make_test_grid_config(0.0, 0.0, 100.0, 100.0, 1.0);

    int col, row;
    bool result = cfg.world_to_cell(-10.0, 50.0, col, row);

    EXPECT_FALSE(result);
}

TEST(GridConfigTest, CellToWorld) {
    auto cfg = make_test_grid_config(0.0, 0.0, 100.0, 100.0, 1.0);

    double wx, wy;
    cfg.cell_to_world(50, 50, wx, wy);

    // Cell center should be at 0.5 offset from cell corner
    EXPECT_TRUE(approx_equal(wx, 50.5));
    EXPECT_TRUE(approx_equal(wy, 49.5));  // negative cell_size_y
}

TEST(GridConfigTest, WorldToCellRoundTrip) {
    auto cfg = make_test_grid_config(0.0, 0.0, 100.0, 100.0, 1.0);

    double wx1 = 25.7;
    double wy1 = 75.3;

    int col, row;
    cfg.world_to_cell(wx1, wy1, col, row);

    double wx2, wy2;
    cfg.cell_to_world(col, row, wx2, wy2);

    // Should be close to original (within cell size)
    EXPECT_LT(std::abs(wx2 - wx1), 1.0);
    EXPECT_LT(std::abs(wy2 - wy1), 1.0);
}

// ===========================================================================
// Tiling Tests
// ===========================================================================

TEST(GridConfigTest, CellToTile) {
    auto cfg = make_test_grid_config(0.0, 0.0, 1000.0, 1000.0, 1.0);
    cfg.tile_width = 256;
    cfg.tile_height = 256;
    cfg.compute_dimensions();

    TileIndex idx = cfg.cell_to_tile(300, 400);

    EXPECT_EQ(idx.col, 1);  // 300 / 256 = 1
    EXPECT_EQ(idx.row, 1);  // 400 / 256 = 1
}

TEST(GridConfigTest, TileBoundsInterior) {
    auto cfg = make_test_grid_config(0.0, 0.0, 1000.0, 1000.0, 1.0);
    cfg.tile_width = 256;
    cfg.tile_height = 256;
    cfg.compute_dimensions();

    TileIndex idx{1, 1};
    BBox bbox = cfg.tile_bounds(idx);

    EXPECT_TRUE(approx_equal(bbox.min_x, 256.0));
    EXPECT_TRUE(approx_equal(bbox.max_x, 512.0));
    EXPECT_TRUE(approx_equal(bbox.max_y, 744.0));  // top edge (negative cell_size_y)
    EXPECT_TRUE(approx_equal(bbox.min_y, 488.0));  // bottom edge
}

TEST(GridConfigTest, TileCellRangeInterior) {
    auto cfg = make_test_grid_config(0.0, 0.0, 1000.0, 1000.0, 1.0);
    cfg.tile_width = 256;
    cfg.tile_height = 256;
    cfg.compute_dimensions();

    TileIndex idx{1, 1};
    int col_start, row_start, col_count, row_count;
    cfg.tile_cell_range(idx, col_start, row_start, col_count, row_count);

    EXPECT_EQ(col_start, 256);
    EXPECT_EQ(row_start, 256);
    EXPECT_EQ(col_count, 256);
    EXPECT_EQ(row_count, 256);
}

TEST(GridConfigTest, TileCellRangeEdge) {
    GridConfig cfg;
    cfg.bounds.min_x = 0.0;
    cfg.bounds.min_y = 0.0;
    cfg.bounds.max_x = 300.0;
    cfg.bounds.max_y = 300.0;
    cfg.cell_size_x = 1.0;
    cfg.cell_size_y = -1.0;
    cfg.tile_width = 256;
    cfg.tile_height = 256;
    cfg.crs = CRS::from_epsg(3857);
    cfg.compute_dimensions();

    // Grid is 300x300, so tiles are 2x2
    // Last tile (1,1) should be smaller
    TileIndex idx{1, 1};
    int col_start, row_start, col_count, row_count;
    cfg.tile_cell_range(idx, col_start, row_start, col_count, row_count);

    EXPECT_EQ(col_start, 256);
    EXPECT_EQ(row_start, 256);
    EXPECT_EQ(col_count, 44);   // 300 - 256 = 44
    EXPECT_EQ(row_count, 44);
}

TEST(GridConfigTest, TotalTiles) {
    auto cfg = make_test_grid_config(0.0, 0.0, 1000.0, 1000.0, 1.0);
    cfg.tile_width = 256;
    cfg.tile_height = 256;
    cfg.compute_dimensions();

    EXPECT_EQ(cfg.total_tiles(), 16);  // 4x4 tiles
}

TEST(GridConfigTest, TotalCells) {
    auto cfg = make_test_grid_config(0.0, 0.0, 1000.0, 1000.0, 1.0);

    EXPECT_EQ(cfg.total_cells(), 1000000);  // 1000x1000
}

// ===========================================================================
// GDAL Geotransform Tests
// ===========================================================================

TEST(GridConfigTest, GDALGeotransform) {
    auto cfg = make_test_grid_config(100.0, 200.0, 1100.0, 1200.0, 10.0);

    double gt[6];
    cfg.gdal_geotransform(gt);

    EXPECT_DOUBLE_EQ(gt[0], 100.0);     // origin_x
    EXPECT_DOUBLE_EQ(gt[1], 10.0);      // cell_size_x
    EXPECT_DOUBLE_EQ(gt[2], 0.0);       // rotation
    EXPECT_DOUBLE_EQ(gt[3], 1200.0);    // origin_y (top edge)
    EXPECT_DOUBLE_EQ(gt[4], 0.0);       // rotation
    EXPECT_DOUBLE_EQ(gt[5], -10.0);     // cell_size_y (negative)
}

// ===========================================================================
// Validation Tests
// ===========================================================================

TEST(GridConfigTest, ValidateValid) {
    auto cfg = make_test_grid_config();

    Status s = cfg.validate();
    EXPECT_TRUE(s.ok());
}

TEST(GridConfigTest, ValidateInvalidBounds) {
    GridConfig cfg;
    cfg.bounds.min_x = 100.0;
    cfg.bounds.max_x = 50.0;  // max < min
    cfg.cell_size_x = 1.0;
    cfg.cell_size_y = -1.0;
    cfg.crs = CRS::from_epsg(3857);

    Status s = cfg.validate();
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST(GridConfigTest, ValidateZeroCellSize) {
    GridConfig cfg;
    cfg.bounds.min_x = 0.0;
    cfg.bounds.max_x = 100.0;
    cfg.bounds.min_y = 0.0;
    cfg.bounds.max_y = 100.0;
    cfg.cell_size_x = 0.0;  // invalid
    cfg.cell_size_y = -1.0;
    cfg.crs = CRS::from_epsg(3857);

    Status s = cfg.validate();
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST(GridConfigTest, ValidateBeforeComputeDimensions) {
    GridConfig cfg;
    cfg.bounds.min_x = 0.0;
    cfg.bounds.max_x = 100.0;
    cfg.bounds.min_y = 0.0;
    cfg.bounds.max_y = 100.0;
    cfg.cell_size_x = 1.0;
    cfg.cell_size_y = -1.0;
    cfg.crs = CRS::from_epsg(3857);
    // Don't call compute_dimensions()

    Status s = cfg.validate();
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST(GridConfigTest, ValidateInvalidCRS) {
    GridConfig cfg;
    cfg.bounds.min_x = 0.0;
    cfg.bounds.max_x = 100.0;
    cfg.bounds.min_y = 0.0;
    cfg.bounds.max_y = 100.0;
    cfg.cell_size_x = 1.0;
    cfg.cell_size_y = -1.0;
    cfg.compute_dimensions();
    // Don't set CRS (invalid)

    Status s = cfg.validate();
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::CrsError);
}
