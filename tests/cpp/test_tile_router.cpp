#include <gtest/gtest.h>
#include "pcr/engine/tile_router.h"
#include "pcr/core/point_cloud.h"
#include "pcr/core/grid_config.h"

using namespace pcr;

class TileRouterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 10x10 grid with 1m resolution, tiled 5x5
        config.bounds = BBox{0.0, 0.0, 10.0, 10.0};
        config.width = 10;
        config.height = 10;
        config.cell_size_x = 1.0;
        config.cell_size_y = -1.0;  // North-up
        config.tile_width = 5;
        config.tile_height = 5;

        // Create point cloud
        cloud = PointCloud::create(100, MemoryLocation::Host);
        cloud->resize(100);

        // Fill with a grid of points
        double* x = const_cast<double*>(cloud->x());
        double* y = const_cast<double*>(cloud->y());

        for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j) {
                int idx = i * 10 + j;
                x[idx] = 0.5 + j;  // Center of cell
                y[idx] = 9.5 - i;  // Center of cell (north-up)
            }
        }

        // Add a value channel
        cloud->add_channel("value", DataType::Float32);
        auto* values = cloud->channel_f32("value");
        for (size_t i = 0; i < 100; ++i) {
            values[i] = static_cast<float>(i);
        }
    }

    GridConfig config;
    std::unique_ptr<PointCloud> cloud;
};

TEST_F(TileRouterTest, Assign_AllPointsValid) {
    auto router = TileRouter::create(config);

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(assignment.num_points, 100);

    // All points should be valid
    for (size_t i = 0; i < assignment.num_points; ++i) {
        EXPECT_EQ(assignment.valid_mask[i], 1);
    }

    // Check cell indices (should be 0..99)
    for (size_t i = 0; i < 100; ++i) {
        int row = i / 10;
        int col = i % 10;
        uint32_t expected_cell = row * 10 + col;
        EXPECT_EQ(assignment.cell_indices[i], expected_cell);
    }

    // Check tile indices
    // Grid is 2x2 tiles (each 5x5)
    for (size_t i = 0; i < 100; ++i) {
        int row = i / 10;
        int col = i % 10;
        int tile_row = row / 5;
        int tile_col = col / 5;
        uint32_t expected_tile = tile_row * 2 + tile_col;
        EXPECT_EQ(assignment.tile_indices[i], expected_tile);
    }

    free(assignment.cell_indices);
    free(assignment.tile_indices);
    free(assignment.valid_mask);
}

TEST_F(TileRouterTest, Assign_OutOfBoundsPoints) {
    // Add some points outside the grid
    auto cloud_oob = PointCloud::create(5, MemoryLocation::Host);
    cloud_oob->resize(5);

    double* x = const_cast<double*>(cloud_oob->x());
    double* y = const_cast<double*>(cloud_oob->y());

    x[0] = -1.0; y[0] = 5.0;  // Outside (left)
    x[1] = 5.0;  y[1] = -1.0; // Outside (bottom)
    x[2] = 15.0; y[2] = 5.0;  // Outside (right)
    x[3] = 5.0;  y[3] = 15.0; // Outside (top)
    x[4] = 5.0;  y[4] = 5.0;  // Inside

    auto router = TileRouter::create(config);

    TileAssignment assignment;
    Status s = router->assign(*cloud_oob, assignment);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(assignment.num_points, 5);

    // First 4 should be invalid
    EXPECT_EQ(assignment.valid_mask[0], 0);
    EXPECT_EQ(assignment.valid_mask[1], 0);
    EXPECT_EQ(assignment.valid_mask[2], 0);
    EXPECT_EQ(assignment.valid_mask[3], 0);

    // Last one should be valid
    EXPECT_EQ(assignment.valid_mask[4], 1);

    free(assignment.cell_indices);
    free(assignment.tile_indices);
    free(assignment.valid_mask);
}

TEST_F(TileRouterTest, Sort_ByTileAndCell) {
    auto router = TileRouter::create(config);

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    // Get values array
    auto* values = cloud->channel_f32("value");
    std::vector<float> values_copy(100);
    memcpy(values_copy.data(), values, 100 * sizeof(float));

    // Sort
    s = router->sort(assignment, values_copy.data(), nullptr, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    // Check that points are sorted by tile first
    uint32_t prev_tile = 0;
    for (size_t i = 0; i < assignment.num_points; ++i) {
        if (!assignment.valid_mask[i]) break;
        EXPECT_GE(assignment.tile_indices[i], prev_tile);
        prev_tile = assignment.tile_indices[i];
    }

    // Within each tile, points should be sorted by cell
    for (uint32_t tile = 0; tile < 4; ++tile) {
        uint32_t prev_cell = 0;
        for (size_t i = 0; i < assignment.num_points; ++i) {
            if (!assignment.valid_mask[i]) break;
            if (assignment.tile_indices[i] == tile) {
                EXPECT_GE(assignment.cell_indices[i], prev_cell);
                prev_cell = assignment.cell_indices[i];
            }
        }
    }

    free(assignment.cell_indices);
    free(assignment.tile_indices);
    free(assignment.valid_mask);
}

TEST_F(TileRouterTest, ExtractBatches_FourTiles) {
    auto router = TileRouter::create(config);

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    auto* values = cloud->channel_f32("value");
    std::vector<float> values_copy(100);
    memcpy(values_copy.data(), values, 100 * sizeof(float));

    s = router->sort(assignment, values_copy.data(), nullptr, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    std::vector<TileBatch> batches;
    s = router->extract_batches(assignment, values_copy.data(), nullptr, nullptr, batches);
    ASSERT_TRUE(s.ok()) << s.message;

    // Should have 4 tiles (2x2 grid of 5x5 tiles)
    EXPECT_EQ(batches.size(), 4);

    // Each tile should have 25 points
    for (const auto& batch : batches) {
        EXPECT_EQ(batch.num_points, 25);

        // Check that local cell indices are in range [0, 25)
        for (size_t i = 0; i < batch.num_points; ++i) {
            EXPECT_LT(batch.local_cell_indices[i], 25);
        }

        // Clean up local cell indices
        free(batch.local_cell_indices);
    }

    free(assignment.cell_indices);
    free(assignment.tile_indices);
    free(assignment.valid_mask);
}

TEST_F(TileRouterTest, ExtractBatches_LocalCellIndices) {
    // Create a simple 4x4 grid with 2x2 tiles
    GridConfig simple_config;
    simple_config.bounds = BBox{0.0, 0.0, 4.0, 4.0};
    simple_config.width = 4;
    simple_config.height = 4;
    simple_config.cell_size_x = 1.0;
    simple_config.cell_size_y = -1.0;
    simple_config.tile_width = 2;
    simple_config.tile_height = 2;

    // Create 4 points, one in each tile
    auto simple_cloud = PointCloud::create(4, MemoryLocation::Host);
    simple_cloud->resize(4);

    double* x = const_cast<double*>(simple_cloud->x());
    double* y = const_cast<double*>(simple_cloud->y());

    // Point in tile (0,0) - top-left
    x[0] = 0.5; y[0] = 3.5;  // Cell (0,0) -> local (0,0) -> index 0

    // Point in tile (0,1) - top-right
    x[1] = 2.5; y[1] = 3.5;  // Cell (0,2) -> local (0,0) -> index 0

    // Point in tile (1,0) - bottom-left
    x[2] = 0.5; y[2] = 1.5;  // Cell (2,0) -> local (0,0) -> index 0

    // Point in tile (1,1) - bottom-right
    x[3] = 2.5; y[3] = 1.5;  // Cell (2,2) -> local (0,0) -> index 0

    simple_cloud->add_channel("value", DataType::Float32);
    auto* values = simple_cloud->channel_f32("value");
    values[0] = 1.0f;
    values[1] = 2.0f;
    values[2] = 3.0f;
    values[3] = 4.0f;

    auto router = TileRouter::create(simple_config);

    TileAssignment assignment;
    Status s = router->assign(*simple_cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    std::vector<float> values_copy(4);
    memcpy(values_copy.data(), values, 4 * sizeof(float));

    s = router->sort(assignment, values_copy.data(), nullptr, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    std::vector<TileBatch> batches;
    s = router->extract_batches(assignment, values_copy.data(), nullptr, nullptr, batches);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(batches.size(), 4);

    // Each batch should have 1 point with local cell index 0
    for (const auto& batch : batches) {
        EXPECT_EQ(batch.num_points, 1);
        EXPECT_EQ(batch.local_cell_indices[0], 0);

        free(batch.local_cell_indices);
    }

    free(assignment.cell_indices);
    free(assignment.tile_indices);
    free(assignment.valid_mask);
}

TEST_F(TileRouterTest, DISABLED_EmptyCloud) {
    auto empty_cloud = PointCloud::create(0, MemoryLocation::Host);
    empty_cloud->resize(0);

    auto router = TileRouter::create(config);

    TileAssignment assignment;
    Status s = router->assign(*empty_cloud, assignment);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(assignment.num_points, 0);
}
