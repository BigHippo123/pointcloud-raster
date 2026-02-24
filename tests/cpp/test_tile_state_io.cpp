#include "pcr/io/tile_state_io.h"
#include "pcr/ops/reduction_registry.h"
#include "test_helpers.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstdio>

using namespace pcr;

// ===========================================================================
// Test Fixture
// ===========================================================================

class TileStateIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/pcr_test_tile_state/";

        // Create test directory
        system(("mkdir -p " + test_dir_).c_str());
    }

    void TearDown() override {
        // Clean up test files
        system(("rm -rf " + test_dir_).c_str());
    }

    std::string test_dir_;
};

// ===========================================================================
// Filename Generation Tests
// ===========================================================================

TEST_F(TileStateIOTest, FilenameGeneration) {
    TileIndex tile{10, 25};
    std::string filename = tile_state_filename("/tmp/tiles", tile);
    EXPECT_EQ(filename, "/tmp/tiles/tile_0010_0025.pcrt");
}

TEST_F(TileStateIOTest, FilenameNoDirectory) {
    TileIndex tile{0, 0};
    std::string filename = tile_state_filename("", tile);
    EXPECT_EQ(filename, "tile_0000_0000.pcrt");
}

TEST_F(TileStateIOTest, FilenameWithTrailingSlash) {
    TileIndex tile{5, 7};
    std::string filename = tile_state_filename("/tmp/tiles/", tile);
    EXPECT_EQ(filename, "/tmp/tiles/tile_0005_0007.pcrt");
}

TEST_F(TileStateIOTest, FilenameLargeIndices) {
    TileIndex tile{9999, 9999};
    std::string filename = tile_state_filename("/tmp", tile);
    EXPECT_EQ(filename, "/tmp/tile_9999_9999.pcrt");
}

// ===========================================================================
// Write/Read Round-Trip Tests
// ===========================================================================

TEST_F(TileStateIOTest, WriteReadRoundTrip) {
    std::string path = test_dir_ + "test.pcrt";

    TileIndex tile_out{5, 10};
    int cols = 256;
    int rows = 256;
    int state_floats = 1;
    ReductionType type = ReductionType::Sum;

    // Create test state data
    std::vector<float> state_out(cols * rows);
    for (int i = 0; i < cols * rows; ++i) {
        state_out[i] = static_cast<float>(i) * 0.5f;
    }

    // Write
    Status s = write_tile_state(path, tile_out, cols, rows, state_floats, type, state_out.data());
    ASSERT_TRUE(s.ok()) << s.message;

    // Read back
    TileIndex tile_in;
    int cols_in, rows_in, state_floats_in;
    ReductionType type_in;
    std::vector<float> state_in(cols * rows);

    s = read_tile_state(path, tile_in, cols_in, rows_in, state_floats_in, type_in, state_in.data());
    ASSERT_TRUE(s.ok()) << s.message;

    // Verify header
    EXPECT_EQ(tile_in.row, 5);
    EXPECT_EQ(tile_in.col, 10);
    EXPECT_EQ(cols_in, 256);
    EXPECT_EQ(rows_in, 256);
    EXPECT_EQ(state_floats_in, 1);
    EXPECT_EQ(type_in, ReductionType::Sum);

    // Verify state data
    for (int i = 0; i < cols * rows; ++i) {
        EXPECT_FLOAT_EQ(state_in[i], state_out[i]);
    }
}

TEST_F(TileStateIOTest, WriteReadMultiStateFloats) {
    std::string path = test_dir_ + "average.pcrt";

    TileIndex tile_out{0, 0};
    int cols = 10;
    int rows = 10;
    int state_floats = 2;  // Average op has 2 state floats (sum, count)
    ReductionType type = ReductionType::Average;

    // Create test state: band-sequential layout
    // sum values: [0..99], count values: [100..199]
    std::vector<float> state_out(state_floats * cols * rows);
    for (int i = 0; i < cols * rows; ++i) {
        state_out[i] = static_cast<float>(i);                  // sum field
        state_out[cols * rows + i] = static_cast<float>(100 + i);  // count field
    }

    // Write
    Status s = write_tile_state(path, tile_out, cols, rows, state_floats, type, state_out.data());
    ASSERT_TRUE(s.ok()) << s.message;

    // Read back
    TileIndex tile_in;
    int cols_in, rows_in, state_floats_in;
    ReductionType type_in;
    std::vector<float> state_in(state_floats * cols * rows);

    s = read_tile_state(path, tile_in, cols_in, rows_in, state_floats_in, type_in, state_in.data());
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(state_floats_in, 2);
    EXPECT_EQ(type_in, ReductionType::Average);

    // Verify state data
    for (int i = 0; i < state_floats * cols * rows; ++i) {
        EXPECT_FLOAT_EQ(state_in[i], state_out[i]);
    }
}

TEST_F(TileStateIOTest, ReadHeaderOnly) {
    std::string path = test_dir_ + "header_test.pcrt";

    TileIndex tile_out{7, 13};
    int cols = 128;
    int rows = 64;
    int state_floats = 1;
    ReductionType type = ReductionType::Max;

    std::vector<float> state_out(cols * rows, 42.0f);

    // Write
    Status s = write_tile_state(path, tile_out, cols, rows, state_floats, type, state_out.data());
    ASSERT_TRUE(s.ok());

    // Read header only
    TileIndex tile_in;
    int cols_in, rows_in, state_floats_in;
    ReductionType type_in;

    s = read_tile_state_header(path, tile_in, cols_in, rows_in, state_floats_in, type_in);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(tile_in.row, 7);
    EXPECT_EQ(tile_in.col, 13);
    EXPECT_EQ(cols_in, 128);
    EXPECT_EQ(rows_in, 64);
    EXPECT_EQ(state_floats_in, 1);
    EXPECT_EQ(type_in, ReductionType::Max);
}

// ===========================================================================
// Error Handling Tests
// ===========================================================================

TEST_F(TileStateIOTest, WriteNullPointer) {
    std::string path = test_dir_ + "null.pcrt";
    TileIndex tile{0, 0};

    Status s = write_tile_state(path, tile, 10, 10, 1, ReductionType::Sum, nullptr);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST_F(TileStateIOTest, WriteInvalidDimensions) {
    std::string path = test_dir_ + "invalid.pcrt";
    TileIndex tile{0, 0};
    std::vector<float> state(100);

    Status s = write_tile_state(path, tile, 0, 10, 1, ReductionType::Sum, state.data());
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);

    s = write_tile_state(path, tile, 10, -5, 1, ReductionType::Sum, state.data());
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);

    s = write_tile_state(path, tile, 10, 10, 0, ReductionType::Sum, state.data());
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST_F(TileStateIOTest, ReadNonexistentFile) {
    std::string path = test_dir_ + "nonexistent.pcrt";

    TileIndex tile;
    int cols, rows, state_floats;
    ReductionType type;
    std::vector<float> state(100);

    Status s = read_tile_state(path, tile, cols, rows, state_floats, type, state.data());
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::IoError);
}

TEST_F(TileStateIOTest, ReadHeaderNonexistentFile) {
    std::string path = test_dir_ + "missing.pcrt";

    TileIndex tile;
    int cols, rows, state_floats;
    ReductionType type;

    Status s = read_tile_state_header(path, tile, cols, rows, state_floats, type);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::IoError);
}

TEST_F(TileStateIOTest, ReadCorruptedFile) {
    std::string path = test_dir_ + "corrupt.pcrt";

    // Write garbage data with wrong magic number
    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);

    // Write enough bytes for a full header, but with wrong magic
    uint32_t bad_magic = 0xDEADBEEF;
    uint8_t buffer[36] = {0};
    memcpy(buffer, &bad_magic, 4);
    fwrite(buffer, 1, sizeof(buffer), f);
    fclose(f);

    TileIndex tile;
    int cols, rows, state_floats;
    ReductionType type;

    Status s = read_tile_state_header(path, tile, cols, rows, state_floats, type);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::IoError);
    // Error could be either "failed to read header" or "invalid magic"
    // Both are valid error cases for a corrupted file
}

TEST_F(TileStateIOTest, ReadNullPointer) {
    std::string path = test_dir_ + "null_read.pcrt";

    // First write a valid file
    TileIndex tile_out{0, 0};
    std::vector<float> state_out(100, 1.0f);
    Status s = write_tile_state(path, tile_out, 10, 10, 1, ReductionType::Sum, state_out.data());
    ASSERT_TRUE(s.ok());

    // Try to read with null pointer
    TileIndex tile_in;
    int cols, rows, state_floats;
    ReductionType type;

    s = read_tile_state(path, tile_in, cols, rows, state_floats, type, nullptr);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

// ===========================================================================
// Real-World Usage Tests
// ===========================================================================

TEST_F(TileStateIOTest, MultipleReductionTypes) {
    struct TestCase {
        ReductionType type;
        int state_floats;
        std::string name;
    };

    std::vector<TestCase> cases = {
        {ReductionType::Sum,     1, "sum"},
        {ReductionType::Max,     1, "max"},
        {ReductionType::Min,     1, "min"},
        {ReductionType::Count,   1, "count"},
        {ReductionType::Average, 2, "avg"},
    };

    for (const auto& tc : cases) {
        std::string path = test_dir_ + tc.name + ".pcrt";
        TileIndex tile{0, 0};
        int cols = 10, rows = 10;

        std::vector<float> state_out(tc.state_floats * cols * rows);
        for (size_t i = 0; i < state_out.size(); ++i) {
            state_out[i] = static_cast<float>(i);
        }

        // Write
        Status s = write_tile_state(path, tile, cols, rows, tc.state_floats, tc.type, state_out.data());
        ASSERT_TRUE(s.ok()) << "Failed to write " << tc.name;

        // Read back
        TileIndex tile_in;
        int cols_in, rows_in, state_floats_in;
        ReductionType type_in;
        std::vector<float> state_in(tc.state_floats * cols * rows);

        s = read_tile_state(path, tile_in, cols_in, rows_in, state_floats_in, type_in, state_in.data());
        ASSERT_TRUE(s.ok()) << "Failed to read " << tc.name;

        EXPECT_EQ(type_in, tc.type);
        EXPECT_EQ(state_floats_in, tc.state_floats);

        for (size_t i = 0; i < state_out.size(); ++i) {
            EXPECT_FLOAT_EQ(state_in[i], state_out[i]) << "Mismatch at index " << i << " for " << tc.name;
        }
    }
}

TEST_F(TileStateIOTest, LargeTileState) {
    std::string path = test_dir_ + "large.pcrt";

    TileIndex tile{100, 200};
    int cols = 512;
    int rows = 512;
    int state_floats = 2;
    ReductionType type = ReductionType::Average;

    // 512*512*2 = 524,288 floats = 2MB
    std::vector<float> state_out(state_floats * cols * rows);
    for (size_t i = 0; i < state_out.size(); ++i) {
        state_out[i] = static_cast<float>(i % 10000) * 0.001f;
    }

    // Write
    Status s = write_tile_state(path, tile, cols, rows, state_floats, type, state_out.data());
    ASSERT_TRUE(s.ok());

    // Read back
    TileIndex tile_in;
    int cols_in, rows_in, state_floats_in;
    ReductionType type_in;
    std::vector<float> state_in(state_floats * cols * rows);

    s = read_tile_state(path, tile_in, cols_in, rows_in, state_floats_in, type_in, state_in.data());
    ASSERT_TRUE(s.ok());

    EXPECT_EQ(tile_in.row, 100);
    EXPECT_EQ(tile_in.col, 200);
    EXPECT_EQ(cols_in, 512);
    EXPECT_EQ(rows_in, 512);

    // Spot check data
    EXPECT_FLOAT_EQ(state_in[0], state_out[0]);
    EXPECT_FLOAT_EQ(state_in[state_in.size() / 2], state_out[state_out.size() / 2]);
    EXPECT_FLOAT_EQ(state_in.back(), state_out.back());
}

TEST_F(TileStateIOTest, SpecialFloatValues) {
    std::string path = test_dir_ + "special.pcrt";

    TileIndex tile{0, 0};
    int cols = 5;
    int rows = 1;
    int state_floats = 1;
    ReductionType type = ReductionType::Sum;

    std::vector<float> state_out = {
        0.0f,
        -0.0f,
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        std::nanf("")
    };

    // Write
    Status s = write_tile_state(path, tile, cols, rows, state_floats, type, state_out.data());
    ASSERT_TRUE(s.ok());

    // Read back
    TileIndex tile_in;
    int cols_in, rows_in, state_floats_in;
    ReductionType type_in;
    std::vector<float> state_in(5);

    s = read_tile_state(path, tile_in, cols_in, rows_in, state_floats_in, type_in, state_in.data());
    ASSERT_TRUE(s.ok());

    EXPECT_FLOAT_EQ(state_in[0], 0.0f);
    EXPECT_FLOAT_EQ(state_in[1], -0.0f);
    EXPECT_TRUE(std::isinf(state_in[2]) && state_in[2] > 0);
    EXPECT_TRUE(std::isinf(state_in[3]) && state_in[3] < 0);
    EXPECT_TRUE(std::isnan(state_in[4]));
}
