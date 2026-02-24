#include "pcr/io/grid_io.h"
#include "pcr/core/grid.h"
#include "test_helpers.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstdio>

using namespace pcr;
using namespace pcr::test;

// ===========================================================================
// Test Fixture
// ===========================================================================

class GridIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/pcr_test_grid_io/";
        system(("mkdir -p " + test_dir_).c_str());
    }

    void TearDown() override {
        system(("rm -rf " + test_dir_).c_str());
    }

    std::string test_dir_;
};

// ===========================================================================
// Write Tests
// ===========================================================================

TEST_F(GridIOTest, WriteSimpleGeoTIFF) {
    std::string path = test_dir_ + "simple.tif";

    // Create a simple 10x10 grid with one band
    auto grid = make_test_grid(10, 10, 1);
    grid->fill(42.0f);

    auto config = make_test_grid_config(0.0, 0.0, 10.0, 10.0, 1.0);

    Status s = write_geotiff(path, *grid, config);
    ASSERT_TRUE(s.ok()) << s.message;

    // Verify file was created
    FILE* f = fopen(path.c_str(), "rb");
    ASSERT_NE(f, nullptr);
    fclose(f);
}

TEST_F(GridIOTest, WriteMultiBandGeoTIFF) {
    std::string path = test_dir_ + "multiband.tif";

    auto grid = make_test_grid(20, 20, 3);
    grid->fill_band(0, 10.0f);
    grid->fill_band(1, 20.0f);
    grid->fill_band(2, 30.0f);

    auto config = make_test_grid_config(100.0, 200.0, 120.0, 220.0, 1.0);

    GeoTiffOptions opts;
    opts.compress = "LZW";

    Status s = write_geotiff(path, *grid, config, opts);
    ASSERT_TRUE(s.ok()) << s.message;
}

TEST_F(GridIOTest, WriteWithDifferentCompression) {
    auto grid = make_test_grid(50, 50, 1);
    grid->fill(123.0f);
    auto config = make_test_grid_config(0.0, 0.0, 50.0, 50.0, 1.0);

    std::vector<std::string> compressors = {"NONE", "LZW", "DEFLATE"};

    for (const auto& comp : compressors) {
        std::string path = test_dir_ + "compress_" + comp + ".tif";
        GeoTiffOptions opts;
        opts.compress = comp;
        opts.compress_level = 6;

        Status s = write_geotiff(path, *grid, config, opts);
        ASSERT_TRUE(s.ok()) << "Failed with compression: " << comp << " - " << s.message;
    }
}

// ===========================================================================
// Read/Write Round-Trip Tests
// ===========================================================================

TEST_F(GridIOTest, RoundTripSingleBand) {
    std::string path = test_dir_ + "roundtrip.tif";

    // Create and write grid
    auto grid_out = make_test_grid(50, 50, 1);
    float* data = grid_out->band_f32(0);
    for (int i = 0; i < 50 * 50; ++i) {
        data[i] = static_cast<float>(i) * 0.5f;
    }

    auto config = make_test_grid_config(0.0, 0.0, 100.0, 100.0, 2.0);
    Status s = write_geotiff(path, *grid_out, config);
    ASSERT_TRUE(s.ok());

    // Read back metadata
    int width, height, num_bands;
    CRS crs;
    BBox bounds;

    s = read_geotiff_info(path, width, height, num_bands, crs, bounds);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(width, 50);
    EXPECT_EQ(height, 50);
    EXPECT_EQ(num_bands, 1);
    EXPECT_TRUE(crs.is_valid());
    EXPECT_DOUBLE_EQ(bounds.min_x, 0.0);
    EXPECT_DOUBLE_EQ(bounds.max_x, 100.0);
    EXPECT_DOUBLE_EQ(bounds.min_y, 0.0);
    EXPECT_DOUBLE_EQ(bounds.max_y, 100.0);

    // Read back data
    std::vector<float> data_in(50 * 50);
    s = read_geotiff_band(path, 0, data_in.data(), 50, 50);
    ASSERT_TRUE(s.ok()) << s.message;

    // Verify data matches
    for (int i = 0; i < 50 * 50; ++i) {
        EXPECT_FLOAT_EQ(data_in[i], data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(GridIOTest, RoundTripMultiBand) {
    std::string path = test_dir_ + "roundtrip_multi.tif";

    // Create 3-band grid
    auto grid_out = make_test_grid(30, 30, 3);

    float* band0 = grid_out->band_f32(0);
    float* band1 = grid_out->band_f32(1);
    float* band2 = grid_out->band_f32(2);

    for (int i = 0; i < 30 * 30; ++i) {
        band0[i] = static_cast<float>(i);
        band1[i] = static_cast<float>(i) * 2.0f;
        band2[i] = static_cast<float>(i) * 3.0f;
    }

    auto config = make_test_grid_config(0.0, 0.0, 30.0, 30.0, 1.0);
    Status s = write_geotiff(path, *grid_out, config);
    ASSERT_TRUE(s.ok());

    // Read back each band
    std::vector<float> data_in(30 * 30);

    for (int b = 0; b < 3; ++b) {
        s = read_geotiff_band(path, b, data_in.data(), 30, 30);
        ASSERT_TRUE(s.ok()) << "Failed to read band " << b;

        const float* expected = grid_out->band_f32(b);
        for (int i = 0; i < 30 * 30; ++i) {
            EXPECT_FLOAT_EQ(data_in[i], expected[i])
                << "Band " << b << " mismatch at index " << i;
        }
    }
}

// ===========================================================================
// TiledGeoTiffWriter Tests
// ===========================================================================

TEST_F(GridIOTest, TiledWriterSingleTile) {
    std::string path = test_dir_ + "tiled_single.tif";

    // Config for 256x256 grid with one 256x256 tile
    GridConfig config;
    config.bounds = {0.0, 0.0, 256.0, 256.0};
    config.cell_size_x = 1.0;
    config.cell_size_y = -1.0;
    config.tile_width = 256;
    config.tile_height = 256;
    config.crs = CRS::from_epsg(3857);
    config.compute_dimensions();

    std::vector<std::string> band_names = {"elevation"};

    auto writer = TiledGeoTiffWriter::open(path, config, band_names);
    ASSERT_NE(writer, nullptr);

    // Create tile data
    std::vector<float> tile_data(256 * 256);
    for (size_t i = 0; i < tile_data.size(); ++i) {
        tile_data[i] = static_cast<float>(i) * 0.1f;
    }

    TileIndex tile{0, 0};
    Status s = writer->write_tile(tile, tile_data.data(), 1);
    ASSERT_TRUE(s.ok()) << s.message;

    s = writer->close();
    ASSERT_TRUE(s.ok()) << s.message;

    // Read back to verify
    int width, height, num_bands;
    CRS crs;
    BBox bounds;
    s = read_geotiff_info(path, width, height, num_bands, crs, bounds);
    ASSERT_TRUE(s.ok());
    EXPECT_EQ(width, 256);
    EXPECT_EQ(height, 256);
    EXPECT_EQ(num_bands, 1);
}

TEST_F(GridIOTest, TiledWriterMultipleTiles) {
    std::string path = test_dir_ + "tiled_multi.tif";

    // Config for 512x512 grid with 2x2 tiles of 256x256 each
    GridConfig config;
    config.bounds = {0.0, 0.0, 512.0, 512.0};
    config.cell_size_x = 1.0;
    config.cell_size_y = -1.0;
    config.tile_width = 256;
    config.tile_height = 256;
    config.crs = CRS::from_epsg(3857);
    config.compute_dimensions();

    std::vector<std::string> band_names = {"sum", "count"};

    auto writer = TiledGeoTiffWriter::open(path, config, band_names);
    ASSERT_NE(writer, nullptr);

    // Write 4 tiles (2x2)
    const int tile_cells = 256 * 256;
    std::vector<float> tile_data(2 * tile_cells);  // 2 bands

    for (int tr = 0; tr < 2; ++tr) {
        for (int tc = 0; tc < 2; ++tc) {
            TileIndex tile{tr, tc};

            // Fill with unique values per tile
            float base = static_cast<float>(tr * 2 + tc) * 1000.0f;
            for (int i = 0; i < tile_cells; ++i) {
                tile_data[i] = base + i;                    // band 0
                tile_data[tile_cells + i] = base + i + 0.5f; // band 1
            }

            Status s = writer->write_tile(tile, tile_data.data(), 2);
            ASSERT_TRUE(s.ok()) << "Failed to write tile (" << tr << ", " << tc << "): " << s.message;
        }
    }

    Status s = writer->close();
    ASSERT_TRUE(s.ok()) << s.message;

    // Verify dimensions
    int width, height, num_bands;
    CRS crs;
    BBox bounds;
    s = read_geotiff_info(path, width, height, num_bands, crs, bounds);
    ASSERT_TRUE(s.ok());
    EXPECT_EQ(width, 512);
    EXPECT_EQ(height, 512);
    EXPECT_EQ(num_bands, 2);
}

TEST_F(GridIOTest, TiledWriterEdgeTile) {
    std::string path = test_dir_ + "tiled_edge.tif";

    // Config for 300x300 grid with tiles of 256x256
    // Last tile will be 44x44
    GridConfig config;
    config.bounds = {0.0, 0.0, 300.0, 300.0};
    config.cell_size_x = 1.0;
    config.cell_size_y = -1.0;
    config.tile_width = 256;
    config.tile_height = 256;
    config.crs = CRS::from_epsg(3857);
    config.compute_dimensions();

    std::vector<std::string> band_names = {"value"};

    auto writer = TiledGeoTiffWriter::open(path, config, band_names);
    ASSERT_NE(writer, nullptr);

    // Write the edge tile (1, 1) which is 44x44
    TileIndex edge_tile{1, 1};
    int col_start, row_start, col_count, row_count;
    config.tile_cell_range(edge_tile, col_start, row_start, col_count, row_count);

    EXPECT_EQ(col_count, 44);
    EXPECT_EQ(row_count, 44);

    std::vector<float> tile_data(col_count * row_count, 99.0f);
    Status s = writer->write_tile(edge_tile, tile_data.data(), 1);
    ASSERT_TRUE(s.ok()) << s.message;

    s = writer->close();
    ASSERT_TRUE(s.ok());
}

// ===========================================================================
// Error Handling Tests
// ===========================================================================

TEST_F(GridIOTest, WriteNonHostGrid) {
    std::string path = test_dir_ + "error.tif";

    // This test would require a GPU grid, which we can't create in CPU-only mode
    // For now, skip this test - it would be tested in GPU mode
}

TEST_F(GridIOTest, WriteDimensionMismatch) {
    std::string path = test_dir_ + "mismatch.tif";

    auto grid = make_test_grid(10, 10, 1);
    auto config = make_test_grid_config(0.0, 0.0, 100.0, 100.0, 1.0);  // 100x100

    Status s = write_geotiff(path, *grid, config);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST_F(GridIOTest, ReadNonexistentFile) {
    std::string path = test_dir_ + "nonexistent.tif";

    int width, height, num_bands;
    CRS crs;
    BBox bounds;

    Status s = read_geotiff_info(path, width, height, num_bands, crs, bounds);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::IoError);
}

TEST_F(GridIOTest, ReadBandNullPointer) {
    std::string path = test_dir_ + "null.tif";

    // First create a valid file
    auto grid = make_test_grid(10, 10, 1);
    auto config = make_test_grid_config(0.0, 0.0, 10.0, 10.0, 1.0);
    Status s = write_geotiff(path, *grid, config);
    ASSERT_TRUE(s.ok()) << s.message;

    // Try to read with null pointer
    s = read_geotiff_band(path, 0, nullptr, 10, 10);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST_F(GridIOTest, ReadBandIndexOutOfRange) {
    std::string path = test_dir_ + "outofrange.tif";

    auto grid = make_test_grid(10, 10, 2);  // 2 bands
    auto config = make_test_grid_config(0.0, 0.0, 10.0, 10.0, 1.0);
    Status s = write_geotiff(path, *grid, config);
    ASSERT_TRUE(s.ok()) << s.message;

    std::vector<float> data(100);
    s = read_geotiff_band(path, 5, data.data(), 10, 10);  // band 5 doesn't exist
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST_F(GridIOTest, ReadDimensionMismatch) {
    std::string path = test_dir_ + "dim_mismatch.tif";

    auto grid = make_test_grid(20, 20, 1);
    auto config = make_test_grid_config(0.0, 0.0, 20.0, 20.0, 1.0);
    Status s = write_geotiff(path, *grid, config);
    ASSERT_TRUE(s.ok());

    std::vector<float> data(100);  // 10x10
    s = read_geotiff_band(path, 0, data.data(), 10, 10);  // wrong size
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST_F(GridIOTest, TiledWriterBandCountMismatch) {
    std::string path = test_dir_ + "band_mismatch.tif";

    auto config = make_test_grid_config();
    config.tile_width = 256;
    config.tile_height = 256;
    config.compute_dimensions();

    std::vector<std::string> band_names = {"a", "b"};  // 2 bands

    auto writer = TiledGeoTiffWriter::open(path, config, band_names);
    ASSERT_NE(writer, nullptr);

    std::vector<float> tile_data(256 * 256);  // only 1 band worth of data
    TileIndex tile{0, 0};

    Status s = writer->write_tile(tile, tile_data.data(), 1);  // wrong band count
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);

    writer->close();
}

// ===========================================================================
// Special Values Tests
// ===========================================================================

TEST_F(GridIOTest, WriteReadNaNValues) {
    std::string path = test_dir_ + "nan.tif";

    auto grid = make_test_grid(5, 5, 1);
    float* data = grid->band_f32(0);

    // Set some values to NaN
    for (int i = 0; i < 25; ++i) {
        data[i] = (i % 2 == 0) ? static_cast<float>(i) : std::nanf("");
    }

    auto config = make_test_grid_config(0.0, 0.0, 5.0, 5.0, 1.0);
    Status s = write_geotiff(path, *grid, config);
    ASSERT_TRUE(s.ok());

    // Read back
    std::vector<float> data_in(25);
    s = read_geotiff_band(path, 0, data_in.data(), 5, 5);
    ASSERT_TRUE(s.ok());

    // Verify NaN values are preserved
    for (int i = 0; i < 25; ++i) {
        if (i % 2 == 0) {
            EXPECT_FLOAT_EQ(data_in[i], static_cast<float>(i));
        } else {
            EXPECT_TRUE(std::isnan(data_in[i])) << "Index " << i << " should be NaN";
        }
    }
}

TEST_F(GridIOTest, WriteReadInfinityValues) {
    std::string path = test_dir_ + "inf.tif";

    auto grid = make_test_grid(4, 1, 1);
    float* data = grid->band_f32(0);

    data[0] = 1.0f;
    data[1] = std::numeric_limits<float>::infinity();
    data[2] = -std::numeric_limits<float>::infinity();
    data[3] = 42.0f;

    auto config = make_test_grid_config(0.0, 0.0, 4.0, 1.0, 1.0);
    Status s = write_geotiff(path, *grid, config);
    ASSERT_TRUE(s.ok());

    // Read back
    std::vector<float> data_in(4);
    s = read_geotiff_band(path, 0, data_in.data(), 4, 1);
    ASSERT_TRUE(s.ok());

    EXPECT_FLOAT_EQ(data_in[0], 1.0f);
    EXPECT_TRUE(std::isinf(data_in[1]) && data_in[1] > 0);
    EXPECT_TRUE(std::isinf(data_in[2]) && data_in[2] < 0);
    EXPECT_FLOAT_EQ(data_in[3], 42.0f);
}

// ===========================================================================
// CRS Tests
// ===========================================================================

TEST_F(GridIOTest, CRSPreservation) {
    std::string path = test_dir_ + "crs.tif";

    auto grid = make_test_grid(10, 10, 1);

    GridConfig config;
    config.bounds = {100000.0, 200000.0, 101000.0, 201000.0};
    config.cell_size_x = 100.0;
    config.cell_size_y = -100.0;
    config.crs = CRS::from_epsg(32610);  // UTM Zone 10N
    config.compute_dimensions();

    Status s = write_geotiff(path, *grid, config);
    ASSERT_TRUE(s.ok());

    // Read back CRS
    int width, height, num_bands;
    CRS crs_in;
    BBox bounds;

    s = read_geotiff_info(path, width, height, num_bands, crs_in, bounds);
    ASSERT_TRUE(s.ok());

    EXPECT_TRUE(crs_in.is_valid());
    EXPECT_EQ(crs_in.epsg, 32610);
}
