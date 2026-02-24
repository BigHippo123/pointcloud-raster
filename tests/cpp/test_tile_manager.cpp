#include <gtest/gtest.h>
#include "pcr/engine/tile_manager.h"
#include "pcr/ops/reduction_registry.h"
#include <filesystem>
#include <cmath>

using namespace pcr;
namespace fs = std::filesystem;

class TileManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory for test
        state_dir = "/tmp/pcr_test_tiles";
        fs::remove_all(state_dir);
        fs::create_directories(state_dir);

        // Create config
        config.state_dir = state_dir;
        config.cache_size_bytes = 1024 * 1024;  // 1MB cache
        config.state_floats = 1;  // Single float per cell (for Sum)

        // Set up grid dimensions
        config.grid_config.width = 100;
        config.grid_config.height = 100;
        config.grid_config.tile_width = 10;
        config.grid_config.tile_height = 10;
        config.grid_config.tiles_x = 10;
        config.grid_config.tiles_y = 10;
    }

    void TearDown() override {
        // Clean up temp directory
        fs::remove_all(state_dir);
    }

    std::string state_dir;
    TileManagerConfig config;
};

TEST_F(TileManagerTest, Create) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    EXPECT_EQ(mgr->tiles_in_cache(), 0);
    EXPECT_EQ(mgr->cache_hits(), 0);
    EXPECT_EQ(mgr->cache_misses(), 0);
}

TEST_F(TileManagerTest, AcquireRelease_SingleTile) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile{0, 0};
    float* state_ptr = nullptr;

    // Acquire tile
    Status s = mgr->acquire(tile, ReductionType::Sum, &state_ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_NE(state_ptr, nullptr);

    EXPECT_EQ(mgr->tiles_in_cache(), 1);
    EXPECT_EQ(mgr->cache_misses(), 1);

    // Check that state is initialized to identity (0 for Sum)
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(state_ptr[i], 0.0f);
    }

    // Modify state
    state_ptr[5] = 42.0f;

    // Release tile
    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(mgr->tiles_in_cache(), 1);  // Still in cache
}

TEST_F(TileManagerTest, AcquireTwice_CacheHit) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile{1, 2};
    float* state_ptr1 = nullptr;
    float* state_ptr2 = nullptr;

    // First acquire (cache miss)
    Status s = mgr->acquire(tile, ReductionType::Sum, &state_ptr1);
    ASSERT_TRUE(s.ok()) << s.message;
    state_ptr1[10] = 123.0f;
    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(mgr->cache_misses(), 1);
    EXPECT_EQ(mgr->cache_hits(), 0);

    // Second acquire (cache hit)
    s = mgr->acquire(tile, ReductionType::Sum, &state_ptr2);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(mgr->cache_misses(), 1);
    EXPECT_EQ(mgr->cache_hits(), 1);

    // Check that state is preserved
    EXPECT_FLOAT_EQ(state_ptr2[10], 123.0f);

    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;
}

TEST_F(TileManagerTest, FlushAll_PersistsToDisk) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile{2, 3};
    float* state_ptr = nullptr;

    // Acquire, modify, release
    Status s = mgr->acquire(tile, ReductionType::Sum, &state_ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    state_ptr[0] = 99.0f;
    state_ptr[50] = 88.0f;
    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    // Flush to disk
    s = mgr->flush_all();
    ASSERT_TRUE(s.ok()) << s.message;

    // Clear cache
    mgr->clear_cache();
    EXPECT_EQ(mgr->tiles_in_cache(), 0);

    // Re-acquire (should load from disk)
    float* state_ptr2 = nullptr;
    s = mgr->acquire(tile, ReductionType::Sum, &state_ptr2);
    ASSERT_TRUE(s.ok()) << s.message;

    // Check that state was loaded correctly
    EXPECT_FLOAT_EQ(state_ptr2[0], 99.0f);
    EXPECT_FLOAT_EQ(state_ptr2[50], 88.0f);

    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;
}

TEST_F(TileManagerTest, MultipleTiles) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    // Acquire multiple tiles
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            TileIndex tile{r, c};
            float* state_ptr = nullptr;

            Status s = mgr->acquire(tile, ReductionType::Sum, &state_ptr);
            ASSERT_TRUE(s.ok()) << s.message;

            // Write unique value
            state_ptr[0] = static_cast<float>(r * 10 + c);

            s = mgr->release(tile);
            ASSERT_TRUE(s.ok()) << s.message;
        }
    }

    EXPECT_EQ(mgr->tiles_in_cache(), 9);

    // Verify values
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            TileIndex tile{r, c};
            float* state_ptr = nullptr;

            Status s = mgr->acquire(tile, ReductionType::Sum, &state_ptr);
            ASSERT_TRUE(s.ok()) << s.message;

            EXPECT_FLOAT_EQ(state_ptr[0], static_cast<float>(r * 10 + c));

            s = mgr->release(tile);
            ASSERT_TRUE(s.ok()) << s.message;
        }
    }

    // All should be cache hits
    EXPECT_EQ(mgr->cache_hits(), 9);
}

TEST_F(TileManagerTest, LRU_Eviction) {
    // Small cache that can only hold 2 tiles
    config.cache_size_bytes = 2 * 100 * sizeof(float);  // 2 tiles worth

    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    // Acquire 3 tiles (should evict oldest)
    TileIndex tile0{0, 0};
    TileIndex tile1{0, 1};
    TileIndex tile2{0, 2};

    float* ptr = nullptr;

    // Acquire tile 0
    Status s = mgr->acquire(tile0, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    ptr[0] = 100.0f;
    s = mgr->release(tile0);
    ASSERT_TRUE(s.ok()) << s.message;

    // Acquire tile 1
    s = mgr->acquire(tile1, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    ptr[0] = 200.0f;
    s = mgr->release(tile1);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(mgr->tiles_in_cache(), 2);

    // Acquire tile 2 (should evict tile 0, the LRU)
    s = mgr->acquire(tile2, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    ptr[0] = 300.0f;
    s = mgr->release(tile2);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(mgr->tiles_in_cache(), 2);

    // Re-acquire tile 0 (cache miss, loaded from disk)
    s = mgr->acquire(tile0, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_FLOAT_EQ(ptr[0], 100.0f);  // Verify it was flushed to disk
    s = mgr->release(tile0);
    ASSERT_TRUE(s.ok()) << s.message;
}

TEST_F(TileManagerTest, TileHasState) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile{5, 5};

    // Initially no state
    EXPECT_FALSE(mgr->tile_has_state(tile));

    // Acquire and release
    float* ptr = nullptr;
    Status s = mgr->acquire(tile, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    // Now has state (in cache)
    EXPECT_TRUE(mgr->tile_has_state(tile));

    // Flush and clear cache
    s = mgr->flush_all();
    ASSERT_TRUE(s.ok()) << s.message;
    mgr->clear_cache();

    // Still has state (on disk)
    EXPECT_TRUE(mgr->tile_has_state(tile));
}

TEST_F(TileManagerTest, DifferentReductionTypes) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile{0, 0};
    float* ptr = nullptr;

    // Acquire with Sum
    Status s = mgr->acquire(tile, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_FLOAT_EQ(ptr[0], 0.0f);  // Sum identity
    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    mgr->clear_cache();

    // Acquire different tile with Max
    config.state_floats = 1;
    auto mgr2 = TileManager::create(config);
    TileIndex tile2{1, 1};

    s = mgr2->acquire(tile2, ReductionType::Max, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_FLOAT_EQ(ptr[0], -FLT_MAX);  // Max identity
    s = mgr2->release(tile2);
    ASSERT_TRUE(s.ok()) << s.message;
}

TEST_F(TileManagerTest, AverageOp_TwoFloatsPerCell) {
    config.state_floats = 2;  // Average needs sum + count

    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile{0, 0};
    float* ptr = nullptr;

    Status s = mgr->acquire(tile, ReductionType::Average, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;

    // State layout: [sum_0, sum_1, ..., sum_99, count_0, count_1, ..., count_99]
    // Check identity (all zeros)
    for (int i = 0; i < 200; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], 0.0f);
    }

    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;
}

TEST_F(TileManagerTest, ClearCache) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    // Acquire some tiles
    for (int i = 0; i < 5; ++i) {
        TileIndex tile{i, 0};
        float* ptr = nullptr;
        Status s = mgr->acquire(tile, ReductionType::Sum, &ptr);
        ASSERT_TRUE(s.ok()) << s.message;
        s = mgr->release(tile);
        ASSERT_TRUE(s.ok()) << s.message;
    }

    EXPECT_EQ(mgr->tiles_in_cache(), 5);

    // Clear cache
    mgr->clear_cache();

    EXPECT_EQ(mgr->tiles_in_cache(), 0);
    EXPECT_EQ(mgr->cache_hits(), 0);
    EXPECT_EQ(mgr->cache_misses(), 0);
}

TEST_F(TileManagerTest, ErrorHandling_NullPointer) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile{0, 0};

    Status s = mgr->acquire(tile, ReductionType::Sum, nullptr);
    EXPECT_FALSE(s.ok());
}

TEST_F(TileManagerTest, ErrorHandling_ReleaseNonAcquired) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile{99, 99};

    Status s = mgr->release(tile);
    EXPECT_FALSE(s.ok());
}

TEST_F(TileManagerTest, PinnedTiles_NotEvicted) {
    // Small cache
    config.cache_size_bytes = 100 * sizeof(float);  // 1 tile

    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    TileIndex tile0{0, 0};
    TileIndex tile1{0, 1};

    float* ptr0 = nullptr;
    float* ptr1 = nullptr;

    // Acquire tile 0 and DON'T release (stays pinned)
    Status s = mgr->acquire(tile0, ReductionType::Sum, &ptr0);
    ASSERT_TRUE(s.ok()) << s.message;

    // Try to acquire tile 1 (should fail because tile 0 is pinned and cache is full)
    s = mgr->acquire(tile1, ReductionType::Sum, &ptr1);
    EXPECT_FALSE(s.ok());  // Can't evict pinned tile

    // Release tile 0
    s = mgr->release(tile0);
    ASSERT_TRUE(s.ok()) << s.message;

    // Now acquire tile 1 should succeed (tile 0 can be evicted)
    s = mgr->acquire(tile1, ReductionType::Sum, &ptr1);
    ASSERT_TRUE(s.ok()) << s.message;
    s = mgr->release(tile1);
    ASSERT_TRUE(s.ok()) << s.message;
}

TEST_F(TileManagerTest, Stats) {
    auto mgr = TileManager::create(config);
    ASSERT_NE(mgr, nullptr);

    EXPECT_EQ(mgr->cache_hits(), 0);
    EXPECT_EQ(mgr->cache_misses(), 0);

    TileIndex tile{0, 0};
    float* ptr = nullptr;

    // First acquire (miss)
    Status s = mgr->acquire(tile, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(mgr->cache_misses(), 1);
    EXPECT_EQ(mgr->cache_hits(), 0);

    // Second acquire (hit)
    s = mgr->acquire(tile, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(mgr->cache_misses(), 1);
    EXPECT_EQ(mgr->cache_hits(), 1);

    // Third acquire (hit)
    s = mgr->acquire(tile, ReductionType::Sum, &ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    s = mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(mgr->cache_misses(), 1);
    EXPECT_EQ(mgr->cache_hits(), 2);
}
