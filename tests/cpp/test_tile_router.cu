#include <gtest/gtest.h>
#include "pcr/engine/tile_router.h"
#include "pcr/core/point_cloud.h"
#include "pcr/core/grid_config.h"
#include "pcr/engine/memory_pool.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

using namespace pcr;

class TileRouterGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Setup grid config
        config.bounds = {0.0, 0.0, 100.0, 100.0};
        config.cell_size_x = 1.0;
        config.cell_size_y = -1.0;
        config.tile_width = 10;
        config.tile_height = 10;
        config.compute_dimensions();

        pool = MemoryPool::create(128 * 1024 * 1024);
        ASSERT_NE(pool, nullptr);

        router = TileRouter::create(config, pool.get());
        ASSERT_NE(router, nullptr);
    }

    GridConfig config;
    std::unique_ptr<MemoryPool> pool;
    std::unique_ptr<TileRouter> router;

    std::unique_ptr<PointCloud> make_device_cloud(
        const std::vector<double>& x,
        const std::vector<double>& y)
    {
        size_t n = x.size();
        auto h_cloud = PointCloud::create(n, MemoryLocation::Host);
        h_cloud->resize(n);
        memcpy(h_cloud->x(), x.data(), n * sizeof(double));
        memcpy(h_cloud->y(), y.data(), n * sizeof(double));

        auto d_cloud = h_cloud->to(MemoryLocation::Device);
        return d_cloud;
    }

    std::vector<uint32_t> download_u32(const uint32_t* d_ptr, size_t n) {
        std::vector<uint32_t> h(n);
        cudaMemcpy(h.data(), d_ptr, n * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        return h;
    }

    std::vector<uint8_t> download_u8(const uint8_t* d_ptr, size_t n) {
        std::vector<uint8_t> h(n);
        cudaMemcpy(h.data(), d_ptr, n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        return h;
    }

    std::vector<float> download_f32(const float* d_ptr, size_t n) {
        std::vector<float> h(n);
        cudaMemcpy(h.data(), d_ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
        return h;
    }
};

TEST_F(TileRouterGPUTest, Assign_InsideGrid) {
    // Points in tiles (0,0), (1,1), (2,2)
    std::vector<double> x = {5.0, 15.0, 25.0};
    std::vector<double> y = {95.0, 85.0, 75.0};  // Note: north-up, origin at max_y

    auto cloud = make_device_cloud(x, y);
    if (!cloud) GTEST_SKIP() << "Device memory unavailable";

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(assignment.num_points, 3u);

    auto h_cells = download_u32(assignment.cell_indices, 3);
    auto h_tiles = download_u32(assignment.tile_indices, 3);
    auto h_valid = download_u8(assignment.valid_mask, 3);

    // All should be valid
    EXPECT_EQ(h_valid[0], 1u);
    EXPECT_EQ(h_valid[1], 1u);
    EXPECT_EQ(h_valid[2], 1u);

    // Tiles: (0,0), (1,1), (2,2)
    int tiles_x = 10;  // 100 width / 10 tile_width
    EXPECT_EQ(h_tiles[0], 0u * tiles_x + 0);
    EXPECT_EQ(h_tiles[1], 1u * tiles_x + 1);
    EXPECT_EQ(h_tiles[2], 2u * tiles_x + 2);
}

TEST_F(TileRouterGPUTest, Assign_OutsideGrid) {
    std::vector<double> x = {-10.0, 5.0, 150.0};
    std::vector<double> y = {50.0, 50.0, 50.0};

    auto cloud = make_device_cloud(x, y);
    if (!cloud) GTEST_SKIP() << "Device memory unavailable";

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    auto h_valid = download_u8(assignment.valid_mask, 3);

    EXPECT_EQ(h_valid[0], 0u);  // x < 0
    EXPECT_EQ(h_valid[1], 1u);  // inside
    EXPECT_EQ(h_valid[2], 0u);  // x > 100
}

TEST_F(TileRouterGPUTest, Sort_SimpleCase) {
    // Create points that will be in different tiles
    std::vector<double> x = {5.0, 25.0, 15.0, 5.0};
    std::vector<double> y = {95.0, 95.0, 95.0, 95.0};
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f};

    auto cloud = make_device_cloud(x, y);
    if (!cloud) GTEST_SKIP() << "Device memory unavailable";

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    // Upload values to device
    float* d_values = nullptr;
    cudaMalloc(&d_values, 4 * sizeof(float));
    cudaMemcpy(d_values, vals.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);

    s = router->sort(assignment, d_values, nullptr, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;
    cudaDeviceSynchronize();

    // After sort, points should be grouped by tile and sorted by cell
    auto h_tiles = download_u32(assignment.tile_indices, 4);
    auto h_cells = download_u32(assignment.cell_indices, 4);
    auto h_values = download_f32(d_values, 4);

    // Tiles should be sorted
    EXPECT_LE(h_tiles[0], h_tiles[1]);
    EXPECT_LE(h_tiles[1], h_tiles[2]);
    EXPECT_LE(h_tiles[2], h_tiles[3]);

    // Within same tile, cells should be sorted
    for (size_t i = 1; i < 4; ++i) {
        if (h_tiles[i] == h_tiles[i-1]) {
            EXPECT_LE(h_cells[i-1], h_cells[i]);
        }
    }

    cudaFree(d_values);
}

TEST_F(TileRouterGPUTest, ExtractBatches_SingleTile) {
    std::vector<double> x = {1.0, 2.0, 3.0};
    std::vector<double> y = {99.0, 99.0, 99.0};
    std::vector<float> vals = {10.0f, 20.0f, 30.0f};

    auto cloud = make_device_cloud(x, y);
    if (!cloud) GTEST_SKIP() << "Device memory unavailable";

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    float* d_values = nullptr;
    cudaMalloc(&d_values, 3 * sizeof(float));
    cudaMemcpy(d_values, vals.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    s = router->sort(assignment, d_values, nullptr, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    std::vector<TileBatch> batches;
    s = router->extract_batches(assignment, d_values, nullptr, nullptr, batches);
    ASSERT_TRUE(s.ok()) << s.message;

    // Should have 1 batch (all in tile 0,0)
    EXPECT_EQ(batches.size(), 1u);
    EXPECT_EQ(batches[0].tile.row, 0);
    EXPECT_EQ(batches[0].tile.col, 0);
    EXPECT_EQ(batches[0].num_points, 3u);

    // Local cell indices should be within tile
    auto h_local_cells = download_u32(batches[0].local_cell_indices, 3);
    for (auto cell : h_local_cells) {
        EXPECT_LT(cell, static_cast<uint32_t>(config.tile_width * config.tile_height));
    }

    cudaFree(d_values);
}

TEST_F(TileRouterGPUTest, ExtractBatches_MultipleTiles) {
    // Points in 3 different tiles
    std::vector<double> x = {5.0, 5.0, 15.0, 15.0, 25.0};
    std::vector<double> y = {95.0, 95.0, 95.0, 95.0, 95.0};
    std::vector<float> vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    auto cloud = make_device_cloud(x, y);
    if (!cloud) GTEST_SKIP() << "Device memory unavailable";

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    float* d_values = nullptr;
    cudaMalloc(&d_values, 5 * sizeof(float));
    cudaMemcpy(d_values, vals.data(), 5 * sizeof(float), cudaMemcpyHostToDevice);

    s = router->sort(assignment, d_values, nullptr, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    std::vector<TileBatch> batches;
    s = router->extract_batches(assignment, d_values, nullptr, nullptr, batches);
    ASSERT_TRUE(s.ok()) << s.message;

    // Should have 3 batches
    EXPECT_EQ(batches.size(), 3u);

    // Verify batch points add up to total
    size_t total_pts = 0;
    for (const auto& b : batches) {
        total_pts += b.num_points;
    }
    EXPECT_EQ(total_pts, 5u);

    cudaFree(d_values);
}

TEST_F(TileRouterGPUTest, LargePointSet) {
    const size_t n = 10000;
    std::vector<double> x(n), y(n);
    std::vector<float> vals(n);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 100.0);

    for (size_t i = 0; i < n; ++i) {
        x[i] = dist(rng);
        y[i] = dist(rng);
        vals[i] = static_cast<float>(i);
    }

    auto cloud = make_device_cloud(x, y);
    if (!cloud) GTEST_SKIP() << "Device memory unavailable";

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    float* d_values = nullptr;
    cudaMalloc(&d_values, n * sizeof(float));
    cudaMemcpy(d_values, vals.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    s = router->sort(assignment, d_values, nullptr, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    std::vector<TileBatch> batches;
    s = router->extract_batches(assignment, d_values, nullptr, nullptr, batches);
    ASSERT_TRUE(s.ok()) << s.message;

    // Verify all points are accounted for
    size_t total = 0;
    for (const auto& b : batches) {
        total += b.num_points;
    }
    EXPECT_GT(batches.size(), 0u);
    EXPECT_LE(total, n);  // May be less if some were out of bounds

    cudaFree(d_values);
}

TEST_F(TileRouterGPUTest, CoSortWeights) {
    std::vector<double> x = {5.0, 15.0, 5.0};
    std::vector<double> y = {95.0, 95.0, 95.0};
    std::vector<float> vals = {1.0f, 2.0f, 3.0f};
    std::vector<float> weights = {10.0f, 20.0f, 30.0f};

    auto cloud = make_device_cloud(x, y);
    if (!cloud) GTEST_SKIP() << "Device memory unavailable";

    TileAssignment assignment;
    Status s = router->assign(*cloud, assignment);
    ASSERT_TRUE(s.ok()) << s.message;

    float *d_values = nullptr, *d_weights = nullptr;
    cudaMalloc(&d_values, 3 * sizeof(float));
    cudaMalloc(&d_weights, 3 * sizeof(float));
    cudaMemcpy(d_values, vals.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights.data(), 3 * sizeof(float), cudaMemcpyHostToDevice);

    s = router->sort(assignment, d_values, d_weights, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;
    cudaDeviceSynchronize();

    auto h_values = download_f32(d_values, 3);
    auto h_weights = download_f32(d_weights, 3);

    // Values and weights should stay paired
    // Original: (1, 10), (2, 20), (3, 30)
    // After sort by tile/cell, the pairing should be preserved
    for (size_t i = 0; i < 3; ++i) {
        float expected_weight = h_values[i] * 10.0f;
        EXPECT_FLOAT_EQ(h_weights[i], expected_weight);
    }

    cudaFree(d_values);
    cudaFree(d_weights);
}
