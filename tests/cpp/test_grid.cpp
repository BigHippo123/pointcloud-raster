#include "pcr/core/grid.h"
#include "test_helpers.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace pcr;
using namespace pcr::test;

// ===========================================================================
// Construction Tests
// ===========================================================================

TEST(GridTest, CreateValid) {
    auto grid = make_test_grid(10, 20, 3);

    ASSERT_NE(grid, nullptr);
    EXPECT_EQ(grid->cols(), 10);
    EXPECT_EQ(grid->rows(), 20);
    EXPECT_EQ(grid->cell_count(), 200);
    EXPECT_EQ(grid->num_bands(), 3);
    EXPECT_EQ(grid->location(), MemoryLocation::Host);
}

TEST(GridTest, CreateInvalidDimensions) {
    std::vector<BandDesc> bands = {{.name = "test", .dtype = DataType::Float32}};

    auto grid1 = Grid::create(0, 10, bands);
    EXPECT_EQ(grid1, nullptr);

    auto grid2 = Grid::create(10, 0, bands);
    EXPECT_EQ(grid2, nullptr);

    auto grid3 = Grid::create(-5, 10, bands);
    EXPECT_EQ(grid3, nullptr);
}

TEST(GridTest, CreateEmptyBands) {
    std::vector<BandDesc> bands;  // empty

    auto grid = Grid::create(10, 10, bands);
    EXPECT_EQ(grid, nullptr);
}

TEST(GridTest, CreateForTile) {
    auto cfg = make_test_grid_config(0.0, 0.0, 1000.0, 1000.0, 1.0);
    cfg.tile_width = 256;
    cfg.tile_height = 256;
    cfg.compute_dimensions();

    std::vector<BandDesc> bands = {{.name = "test", .dtype = DataType::Float32}};
    TileIndex idx{0, 0};

    auto grid = Grid::create_for_tile(cfg, idx, bands);

    ASSERT_NE(grid, nullptr);
    EXPECT_EQ(grid->cols(), 256);
    EXPECT_EQ(grid->rows(), 256);
}

TEST(GridTest, CreateForEdgeTile) {
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

    std::vector<BandDesc> bands = {{.name = "test", .dtype = DataType::Float32}};
    TileIndex idx{1, 1};  // edge tile

    auto grid = Grid::create_for_tile(cfg, idx, bands);

    ASSERT_NE(grid, nullptr);
    EXPECT_EQ(grid->cols(), 44);   // 300 - 256
    EXPECT_EQ(grid->rows(), 44);
}

// ===========================================================================
// Band Management Tests
// ===========================================================================

TEST(GridTest, NumBands) {
    auto grid = make_test_grid(10, 10, 5);
    EXPECT_EQ(grid->num_bands(), 5);
}

TEST(GridTest, BandDescValid) {
    auto grid = make_test_grid(10, 10, 3);

    BandDesc desc = grid->band_desc(0);
    EXPECT_EQ(desc.name, "band0");
    EXPECT_EQ(desc.dtype, DataType::Float32);

    desc = grid->band_desc(2);
    EXPECT_EQ(desc.name, "band2");
}

TEST(GridTest, BandDescInvalid) {
    auto grid = make_test_grid(10, 10, 2);

    BandDesc desc = grid->band_desc(5);  // out of range
    EXPECT_TRUE(desc.name.empty());

    desc = grid->band_desc(-1);
    EXPECT_TRUE(desc.name.empty());
}

TEST(GridTest, BandIndexByName) {
    auto grid = make_test_grid(10, 10, 3);

    EXPECT_EQ(grid->band_index("band0"), 0);
    EXPECT_EQ(grid->band_index("band1"), 1);
    EXPECT_EQ(grid->band_index("band2"), 2);
    EXPECT_EQ(grid->band_index("nonexistent"), -1);
}

TEST(GridTest, BandDataAccessors) {
    auto grid = make_test_grid(10, 10, 2);

    void* data0 = grid->band_data(0);
    EXPECT_NE(data0, nullptr);

    void* data1 = grid->band_data(1);
    EXPECT_NE(data1, nullptr);
    EXPECT_NE(data0, data1);  // different bands

    void* data_invalid = grid->band_data(5);
    EXPECT_EQ(data_invalid, nullptr);
}

TEST(GridTest, BandF32Accessors) {
    auto grid = make_test_grid(10, 10, 2);

    float* data0 = grid->band_f32(0);
    EXPECT_NE(data0, nullptr);

    float* data_by_name = grid->band_f32("band1");
    EXPECT_NE(data_by_name, nullptr);

    float* data_invalid = grid->band_f32("nonexistent");
    EXPECT_EQ(data_invalid, nullptr);
}

// ===========================================================================
// Data Operations Tests
// ===========================================================================

TEST(GridTest, FillAllBands) {
    auto grid = make_test_grid(10, 10, 2);

    Status s = grid->fill(42.0f);
    EXPECT_TRUE(s.ok());

    // Verify values
    const float* data0 = grid->band_f32(0);
    const float* data1 = grid->band_f32(1);

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(data0[i], 42.0f);
        EXPECT_FLOAT_EQ(data1[i], 42.0f);
    }
}

TEST(GridTest, FillSingleBand) {
    auto grid = make_test_grid(10, 10, 2);

    grid->fill(0.0f);
    Status s = grid->fill_band(0, 99.0f);
    EXPECT_TRUE(s.ok());

    const float* data0 = grid->band_f32(0);
    const float* data1 = grid->band_f32(1);

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(data0[i], 99.0f);
        EXPECT_FLOAT_EQ(data1[i], 0.0f);
    }
}

TEST(GridTest, FillBandInvalid) {
    auto grid = make_test_grid(10, 10, 2);

    Status s = grid->fill_band(5, 1.0f);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

// ===========================================================================
// Memory Operations Tests
// ===========================================================================

TEST(GridTest, CopyFromSameDimensions) {
    auto grid1 = make_test_grid(10, 10, 2);
    auto grid2 = make_test_grid(10, 10, 2);

    // Fill grid1 with test data
    grid1->fill_band(0, 10.0f);
    grid1->fill_band(1, 20.0f);

    // Copy to grid2
    Status s = grid2->copy_from(*grid1);
    EXPECT_TRUE(s.ok());

    // Verify copied data
    const float* data0 = grid2->band_f32(0);
    const float* data1 = grid2->band_f32(1);

    EXPECT_FLOAT_EQ(data0[0], 10.0f);
    EXPECT_FLOAT_EQ(data1[0], 20.0f);
}

TEST(GridTest, CopyFromDimensionMismatch) {
    auto grid1 = make_test_grid(10, 10, 2);
    auto grid2 = make_test_grid(20, 20, 2);

    Status s = grid2->copy_from(*grid1);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST(GridTest, CopyFromBandCountMismatch) {
    auto grid1 = make_test_grid(10, 10, 2);
    auto grid2 = make_test_grid(10, 10, 3);

    Status s = grid2->copy_from(*grid1);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST(GridTest, ToSameLocation) {
    auto grid1 = make_test_grid(10, 10, 2);
    grid1->fill_band(0, 123.0f);

    auto grid2 = grid1->to(MemoryLocation::Host);

    ASSERT_NE(grid2, nullptr);
    EXPECT_EQ(grid2->cols(), 10);
    EXPECT_EQ(grid2->rows(), 10);
    EXPECT_EQ(grid2->num_bands(), 2);

    const float* data = grid2->band_f32(0);
    EXPECT_FLOAT_EQ(data[0], 123.0f);
}

// ===========================================================================
// Valid Mask Tests
// ===========================================================================

TEST(GridTest, ValidMaskFiniteValues) {
    auto grid = make_test_grid(5, 5, 1);
    grid->fill(42.0f);

    std::vector<uint8_t> mask = grid->valid_mask(0);

    ASSERT_EQ(mask.size(), 25);
    for (uint8_t v : mask) {
        EXPECT_EQ(v, 1);
    }
}

TEST(GridTest, ValidMaskWithNaN) {
    auto grid = make_test_grid(5, 5, 1);

    float* data = grid->band_f32(0);
    for (int i = 0; i < 25; ++i) {
        data[i] = (i % 2 == 0) ? 1.0f : std::nanf("");
    }

    std::vector<uint8_t> mask = grid->valid_mask(0);

    ASSERT_EQ(mask.size(), 25);
    for (int i = 0; i < 25; ++i) {
        EXPECT_EQ(mask[i], (i % 2 == 0) ? 1 : 0);
    }
}

TEST(GridTest, ValidMaskWithInf) {
    auto grid = make_test_grid(5, 5, 1);

    float* data = grid->band_f32(0);
    data[0] = 1.0f;
    data[1] = std::numeric_limits<float>::infinity();
    data[2] = -std::numeric_limits<float>::infinity();
    data[3] = std::nanf("");

    std::vector<uint8_t> mask = grid->valid_mask(0);

    EXPECT_EQ(mask[0], 1);  // finite
    EXPECT_EQ(mask[1], 0);  // +inf
    EXPECT_EQ(mask[2], 0);  // -inf
    EXPECT_EQ(mask[3], 0);  // nan
}

// ===========================================================================
// Edge Cases Tests
// ===========================================================================

TEST(GridTest, SingleCellGrid) {
    std::vector<BandDesc> bands = {{.name = "test", .dtype = DataType::Float32}};
    auto grid = Grid::create(1, 1, bands);

    ASSERT_NE(grid, nullptr);
    EXPECT_EQ(grid->cols(), 1);
    EXPECT_EQ(grid->rows(), 1);
    EXPECT_EQ(grid->cell_count(), 1);

    Status s = grid->fill(99.0f);
    EXPECT_TRUE(s.ok());

    const float* data = grid->band_f32(0);
    EXPECT_FLOAT_EQ(data[0], 99.0f);
}

TEST(GridTest, ManyBands) {
    auto grid = make_test_grid(10, 10, 15);

    EXPECT_EQ(grid->num_bands(), 15);

    for (int i = 0; i < 15; ++i) {
        EXPECT_NE(grid->band_f32(i), nullptr);
    }
}

TEST(GridTest, LargeGrid) {
    // 1000x1000 = 1M cells
    auto grid = make_test_grid(1000, 1000, 1);

    ASSERT_NE(grid, nullptr);
    EXPECT_EQ(grid->cell_count(), 1000000);

    Status s = grid->fill(1.0f);
    EXPECT_TRUE(s.ok());
}
