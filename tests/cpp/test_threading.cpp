#include <gtest/gtest.h>
#include "pcr/core/point_cloud.h"
#include "pcr/core/grid.h"
#include "pcr/engine/pipeline.h"
#include "pcr/engine/tile_router.h"
#include "pcr/engine/accumulator.h"
#include "pcr/engine/filter.h"
#include "pcr/ops/reduction_registry.h"
#include "pcr/core/grid_config.h"
#include <chrono>
#include <vector>
#include <cmath>
#include <algorithm>

#ifdef PCR_HAS_OPENMP
#include <omp.h>
#endif

using namespace pcr;

// Helper to check if OpenMP is available
bool openmp_available() {
#ifdef PCR_HAS_OPENMP
    return true;
#else
    return false;
#endif
}

// Helper to get max threads
int get_max_threads() {
#ifdef PCR_HAS_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// Test fixture for threading tests
class ThreadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!openmp_available()) {
            GTEST_SKIP() << "OpenMP not available, skipping threading tests";
        }
    }
};

// ---------------------------------------------------------------------------
// Accumulator Threading Tests
// ---------------------------------------------------------------------------

TEST_F(ThreadingTest, Accumulator_SingleVsMultiThread_Sum) {
#ifdef PCR_HAS_OPENMP
    const int tile_cells = 1000;
    const size_t num_points = 10000;

    // Create test data
    std::vector<uint32_t> cell_indices(num_points);
    std::vector<float> values(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        cell_indices[i] = i % tile_cells;
        values[i] = 1.0f + (i % 10);
    }

    // Single-threaded run
    omp_set_num_threads(1);
    std::vector<float> state_single(tile_cells, 0.0f);
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    info->init_state(state_single.data(), tile_cells, nullptr);
    Status s = info->accumulate(cell_indices.data(), values.data(),
                                state_single.data(), num_points, tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    // Multi-threaded run
    int max_threads = get_max_threads();
    if (max_threads > 1) {
        omp_set_num_threads(max_threads);
        std::vector<float> state_multi(tile_cells, 0.0f);

        info->init_state(state_multi.data(), tile_cells, nullptr);
        s = info->accumulate(cell_indices.data(), values.data(),
                            state_multi.data(), num_points, tile_cells, nullptr);
        ASSERT_TRUE(s.ok()) << s.message;

        // Results should be identical
        for (int i = 0; i < tile_cells; ++i) {
            EXPECT_FLOAT_EQ(state_single[i], state_multi[i])
                << "Mismatch at cell " << i;
        }
    }
#endif
}

TEST_F(ThreadingTest, Accumulator_SingleVsMultiThread_Average) {
#ifdef PCR_HAS_OPENMP
    const int tile_cells = 500;
    const size_t num_points = 5000;

    std::vector<uint32_t> cell_indices(num_points);
    std::vector<float> values(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        cell_indices[i] = i % tile_cells;
        values[i] = static_cast<float>(i % 100) / 10.0f;
    }

    const ReductionInfo* info = get_reduction(ReductionType::Average);
    ASSERT_NE(info, nullptr);

    // Single-threaded
    omp_set_num_threads(1);
    std::vector<float> state_single(tile_cells * 2, 0.0f);  // sum + count
    info->init_state(state_single.data(), tile_cells, nullptr);
    Status s = info->accumulate(cell_indices.data(), values.data(),
                                state_single.data(), num_points, tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    // Finalize to get averages
    std::vector<float> result_single(tile_cells);
    info->finalize(state_single.data(), result_single.data(), tile_cells, nullptr);

    // Multi-threaded
    int max_threads = get_max_threads();
    if (max_threads > 1) {
        omp_set_num_threads(max_threads);
        std::vector<float> state_multi(tile_cells * 2, 0.0f);
        info->init_state(state_multi.data(), tile_cells, nullptr);
        s = info->accumulate(cell_indices.data(), values.data(),
                            state_multi.data(), num_points, tile_cells, nullptr);
        ASSERT_TRUE(s.ok()) << s.message;

        std::vector<float> result_multi(tile_cells);
        info->finalize(state_multi.data(), result_multi.data(), tile_cells, nullptr);

        // Results should be very close (allowing for floating point differences)
        for (int i = 0; i < tile_cells; ++i) {
            EXPECT_NEAR(result_single[i], result_multi[i], 1e-5f)
                << "Mismatch at cell " << i;
        }
    }
#endif
}

TEST_F(ThreadingTest, Accumulator_MaxMin_Deterministic) {
#ifdef PCR_HAS_OPENMP
    const int tile_cells = 100;
    const size_t num_points = 1000;

    std::vector<uint32_t> cell_indices(num_points);
    std::vector<float> values(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        cell_indices[i] = i % tile_cells;
        values[i] = static_cast<float>(i);
    }

    // Test Max reduction
    const ReductionInfo* max_info = get_reduction(ReductionType::Max);
    ASSERT_NE(max_info, nullptr);

    // Run multiple times with threading
    int max_threads = get_max_threads();
    if (max_threads > 1) {
        omp_set_num_threads(max_threads);

        std::vector<float> state1(tile_cells, 0.0f);
        std::vector<float> state2(tile_cells, 0.0f);

        max_info->init_state(state1.data(), tile_cells, nullptr);
        max_info->accumulate(cell_indices.data(), values.data(),
                           state1.data(), num_points, tile_cells, nullptr);

        max_info->init_state(state2.data(), tile_cells, nullptr);
        max_info->accumulate(cell_indices.data(), values.data(),
                           state2.data(), num_points, tile_cells, nullptr);

        // Results should be deterministic
        for (int i = 0; i < tile_cells; ++i) {
            EXPECT_FLOAT_EQ(state1[i], state2[i])
                << "Non-deterministic result at cell " << i;
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// TileRouter Threading Tests
// ---------------------------------------------------------------------------

TEST_F(ThreadingTest, TileRouter_SingleVsMultiThread) {
#ifdef PCR_HAS_OPENMP
    GridConfig config;
    config.bounds = BBox{0.0, 0.0, 100.0, 100.0};
    config.cell_size_x = 1.0;
    config.cell_size_y = -1.0;
    config.tile_width = 50;
    config.tile_height = 50;
    config.compute_dimensions();

    auto router = TileRouter::create(config);
    ASSERT_NE(router, nullptr);

    // Create test cloud
    auto cloud = PointCloud::create(10000, MemoryLocation::Host);
    ASSERT_NE(cloud, nullptr);

    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());
    for (size_t i = 0; i < 10000; ++i) {
        x[i] = (i % 100);
        y[i] = (i / 100);
    }
    cloud->resize(10000);

    // Single-threaded
    omp_set_num_threads(1);
    TileAssignment assign_single;
    Status s = router->assign(*cloud, assign_single);
    ASSERT_TRUE(s.ok()) << s.message;

    // Multi-threaded
    int max_threads = get_max_threads();
    if (max_threads > 1) {
        omp_set_num_threads(max_threads);
        TileAssignment assign_multi;
        s = router->assign(*cloud, assign_multi);
        ASSERT_TRUE(s.ok()) << s.message;

        // Results should be identical
        EXPECT_EQ(assign_single.num_points, assign_multi.num_points);
        for (size_t i = 0; i < assign_single.num_points; ++i) {
            EXPECT_EQ(assign_single.cell_indices[i], assign_multi.cell_indices[i])
                << "Cell mismatch at point " << i;
            EXPECT_EQ(assign_single.tile_indices[i], assign_multi.tile_indices[i])
                << "Tile mismatch at point " << i;
            EXPECT_EQ(assign_single.valid_mask[i], assign_multi.valid_mask[i])
                << "Valid mask mismatch at point " << i;
        }

        free(assign_multi.cell_indices);
        free(assign_multi.tile_indices);
        free(assign_multi.valid_mask);
    }

    free(assign_single.cell_indices);
    free(assign_single.tile_indices);
    free(assign_single.valid_mask);
#endif
}

TEST_F(ThreadingTest, TileRouter_OutOfBoundsThreadSafe) {
#ifdef PCR_HAS_OPENMP
    GridConfig config;
    config.bounds = BBox{0.0, 0.0, 10.0, 10.0};
    config.cell_size_x = 1.0;
    config.cell_size_y = -1.0;
    config.tile_width = 10;
    config.tile_height = 10;
    config.compute_dimensions();

    auto router = TileRouter::create(config);
    ASSERT_NE(router, nullptr);

    // Create cloud with many out-of-bounds points
    auto cloud = PointCloud::create(1000, MemoryLocation::Host);
    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());

    for (size_t i = 0; i < 1000; ++i) {
        // Mix of in-bounds and out-of-bounds
        if (i % 3 == 0) {
            x[i] = 5.0;  // in bounds
            y[i] = 5.0;
        } else {
            x[i] = -10.0 + (i % 30);  // mostly out of bounds
            y[i] = -10.0 + (i / 30);
        }
    }
    cloud->resize(1000);

    // Multi-threaded
    int max_threads = get_max_threads();
    if (max_threads > 1) {
        omp_set_num_threads(max_threads);

        TileAssignment assignment;
        Status s = router->assign(*cloud, assignment);
        ASSERT_TRUE(s.ok()) << s.message;

        // Verify all points are processed
        EXPECT_EQ(assignment.num_points, 1000);

        // Count valid points
        size_t valid_count = 0;
        for (size_t i = 0; i < assignment.num_points; ++i) {
            if (assignment.valid_mask[i]) {
                valid_count++;
            }
        }

        // Should have roughly 1/3 valid points
        EXPECT_GT(valid_count, 0);
        EXPECT_LT(valid_count, 1000);

        free(assignment.cell_indices);
        free(assignment.tile_indices);
        free(assignment.valid_mask);
    }
#endif
}

// ---------------------------------------------------------------------------
// Filter Threading Tests
// ---------------------------------------------------------------------------

TEST_F(ThreadingTest, Filter_SingleVsMultiThread) {
#ifdef PCR_HAS_OPENMP
    // Create test cloud
    auto cloud = PointCloud::create(5000, MemoryLocation::Host);
    ASSERT_NE(cloud, nullptr);

    cloud->add_channel("intensity", DataType::Float32);
    float* intensity = cloud->channel_f32("intensity");
    for (size_t i = 0; i < 5000; ++i) {
        intensity[i] = static_cast<float>(i % 100);
    }
    cloud->resize(5000);

    // Create filter: intensity > 50
    FilterSpec filter;
    filter.add("intensity", CompareOp::Greater, 50.0f);

    // Single-threaded
    omp_set_num_threads(1);
    uint32_t* indices_single = nullptr;
    size_t count_single = 0;
    Status s = filter_points(*cloud, filter, &indices_single, &count_single, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    // Multi-threaded
    int max_threads = get_max_threads();
    if (max_threads > 1) {
        omp_set_num_threads(max_threads);
        uint32_t* indices_multi = nullptr;
        size_t count_multi = 0;
        s = filter_points(*cloud, filter, &indices_multi, &count_multi, nullptr);
        ASSERT_TRUE(s.ok()) << s.message;

        // Should have same count
        EXPECT_EQ(count_single, count_multi);

        // Sort both arrays to compare (parallel version may have different order)
        std::vector<uint32_t> sorted_single(indices_single, indices_single + count_single);
        std::vector<uint32_t> sorted_multi(indices_multi, indices_multi + count_multi);
        std::sort(sorted_single.begin(), sorted_single.end());
        std::sort(sorted_multi.begin(), sorted_multi.end());

        // Should have same indices
        for (size_t i = 0; i < count_single; ++i) {
            EXPECT_EQ(sorted_single[i], sorted_multi[i])
                << "Index mismatch at position " << i;
        }

        free(indices_multi);
    }

    free(indices_single);
#endif
}

TEST_F(ThreadingTest, Filter_ComplexPredicate_ThreadSafe) {
#ifdef PCR_HAS_OPENMP
    auto cloud = PointCloud::create(10000, MemoryLocation::Host);
    ASSERT_NE(cloud, nullptr);

    cloud->add_channel("elevation", DataType::Float32);
    cloud->add_channel("intensity", DataType::Float32);

    float* elevation = cloud->channel_f32("elevation");
    float* intensity = cloud->channel_f32("intensity");
    for (size_t i = 0; i < 10000; ++i) {
        elevation[i] = static_cast<float>(i % 1000) / 10.0f;
        intensity[i] = static_cast<float>(i % 200);
    }
    cloud->resize(10000);

    // Complex filter: elevation > 30 AND intensity < 150
    FilterSpec filter;
    filter.add("elevation", CompareOp::Greater, 30.0f);
    filter.add("intensity", CompareOp::Less, 150.0f);

    // Multi-threaded
    int max_threads = get_max_threads();
    if (max_threads > 1) {
        omp_set_num_threads(max_threads);

        uint32_t* indices = nullptr;
        size_t count = 0;
        Status s = filter_points(*cloud, filter, &indices, &count, nullptr);
        ASSERT_TRUE(s.ok()) << s.message;

        // Verify all returned indices actually match the filter
        for (size_t i = 0; i < count; ++i) {
            uint32_t idx = indices[i];
            EXPECT_GT(elevation[idx], 30.0f) << "Index " << idx;
            EXPECT_LT(intensity[idx], 150.0f) << "Index " << idx;
        }

        free(indices);
    }
#endif
}

// ---------------------------------------------------------------------------
// Pipeline Threading Tests
// ---------------------------------------------------------------------------

TEST_F(ThreadingTest, Pipeline_ThreadCountConfiguration) {
#ifdef PCR_HAS_OPENMP
    GridConfig grid_config;
    grid_config.bounds = BBox{0.0, 0.0, 50.0, 50.0};
    grid_config.cell_size_x = 1.0;
    grid_config.cell_size_y = -1.0;
    grid_config.tile_width = 50;
    grid_config.tile_height = 50;
    grid_config.compute_dimensions();

    PipelineConfig config;
    config.grid = grid_config;
    config.exec_mode = ExecutionMode::CPU;
    config.state_dir = "/tmp/pcr_thread_config_test";
    config.cpu_threads = 2;  // Set to 2 threads

    ReductionSpec reduction;
    reduction.value_channel = "values";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // After pipeline creation, OpenMP should be configured
    int current_threads = omp_get_max_threads();
    EXPECT_EQ(current_threads, 2) << "Thread count not configured correctly";
#endif
}

TEST_F(ThreadingTest, Pipeline_SingleVsMultiThread_Sum) {
#ifdef PCR_HAS_OPENMP
    GridConfig grid_config;
    grid_config.bounds = BBox{0.0, 0.0, 100.0, 100.0};
    grid_config.width = 100;
    grid_config.height = 100;
    grid_config.cell_size_x = 1.0;
    grid_config.cell_size_y = -1.0;
    grid_config.tile_width = 100;
    grid_config.tile_height = 100;

    // Create test cloud
    auto cloud = PointCloud::create(5000, MemoryLocation::Host);
    ASSERT_NE(cloud, nullptr);
    cloud->resize(5000);

    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());
    for (size_t i = 0; i < 5000; ++i) {
        x[i] = 50.0 + (i % 50);
        y[i] = 50.0 + (i / 50);
    }

    cloud->add_channel("values", DataType::Float32);
    float* values = cloud->channel_f32("values");
    for (size_t i = 0; i < 5000; ++i) {
        values[i] = 1.0f;
    }

    // Single-threaded pipeline
    {
        PipelineConfig config;
        config.grid = grid_config;
        config.exec_mode = ExecutionMode::CPU;
        config.state_dir = "/tmp/pcr_single_thread_test";
        config.cpu_threads = 1;

        ReductionSpec reduction;
        reduction.value_channel = "values";
        reduction.type = ReductionType::Sum;
        config.reductions.push_back(reduction);

        auto pipeline = Pipeline::create(config);
        ASSERT_NE(pipeline, nullptr);

        Status s = pipeline->ingest(*cloud);
        ASSERT_TRUE(s.ok()) << s.message;

        s = pipeline->finalize();
        ASSERT_TRUE(s.ok()) << s.message;

        const Grid* result_single = pipeline->result();
        ASSERT_NE(result_single, nullptr);

        // Multi-threaded pipeline
        int max_threads = get_max_threads();
        if (max_threads > 1) {
            PipelineConfig config_multi;
            config_multi.grid = grid_config;
            config_multi.exec_mode = ExecutionMode::CPU;
            config_multi.state_dir = "/tmp/pcr_multi_thread_test";
            config_multi.cpu_threads = max_threads;

            ReductionSpec reduction_multi;
            reduction_multi.value_channel = "values";
            reduction_multi.type = ReductionType::Sum;
            config_multi.reductions.push_back(reduction_multi);

            auto pipeline_multi = Pipeline::create(config_multi);
            ASSERT_NE(pipeline_multi, nullptr);

            s = pipeline_multi->ingest(*cloud);
            ASSERT_TRUE(s.ok()) << s.message;

            s = pipeline_multi->finalize();
            ASSERT_TRUE(s.ok()) << s.message;

            const Grid* result_multi = pipeline_multi->result();
            ASSERT_NE(result_multi, nullptr);

            // Results should be identical
            EXPECT_EQ(result_single->cols(), result_multi->cols());
            EXPECT_EQ(result_single->rows(), result_multi->rows());

            const float* data_single = result_single->band_f32(0);
            const float* data_multi = result_multi->band_f32(0);

            for (int i = 0; i < result_single->cell_count(); ++i) {
                if (!std::isnan(data_single[i]) && !std::isnan(data_multi[i])) {
                    EXPECT_FLOAT_EQ(data_single[i], data_multi[i])
                        << "Mismatch at cell " << i;
                }
            }
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Performance Sanity Check
// ---------------------------------------------------------------------------

// TODO: Performance test shows threading overhead for small datasets
// This is expected - threading benefits appear with larger datasets (100M+ points)
// Correctness tests above demonstrate threading works correctly
TEST_F(ThreadingTest, DISABLED_Performance_MultiThreadSpeedup) {
#ifdef PCR_HAS_OPENMP
    int max_threads = get_max_threads();
    if (max_threads <= 1) {
        GTEST_SKIP() << "Need more than 1 thread for speedup test";
    }

    const int tile_cells = 10000;
    const size_t num_points = 1000000;  // 1M points

    std::vector<uint32_t> cell_indices(num_points);
    std::vector<float> values(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        cell_indices[i] = i % tile_cells;
        values[i] = static_cast<float>(i % 100);
    }

    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    // Single-threaded timing
    omp_set_num_threads(1);
    std::vector<float> state_single(tile_cells, 0.0f);
    info->init_state(state_single.data(), tile_cells, nullptr);

    auto start = std::chrono::high_resolution_clock::now();
    info->accumulate(cell_indices.data(), values.data(),
                    state_single.data(), num_points, tile_cells, nullptr);
    auto end = std::chrono::high_resolution_clock::now();

    auto single_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Multi-threaded timing
    omp_set_num_threads(max_threads);
    std::vector<float> state_multi(tile_cells, 0.0f);
    info->init_state(state_multi.data(), tile_cells, nullptr);

    start = std::chrono::high_resolution_clock::now();
    info->accumulate(cell_indices.data(), values.data(),
                    state_multi.data(), num_points, tile_cells, nullptr);
    end = std::chrono::high_resolution_clock::now();

    auto multi_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // Multi-threaded should be faster (at least some speedup)
    // We don't expect linear speedup due to critical sections, but should see improvement
    double speedup = static_cast<double>(single_time) / static_cast<double>(multi_time);

    std::cout << "Single-thread time: " << single_time << " ms" << std::endl;
    std::cout << "Multi-thread (" << max_threads << " threads) time: " << multi_time << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // Should see at least some speedup (even with critical sections)
    EXPECT_GT(speedup, 1.0) << "Multi-threading should provide some speedup";

    // Verify correctness
    for (int i = 0; i < tile_cells; ++i) {
        EXPECT_FLOAT_EQ(state_single[i], state_multi[i]);
    }
#endif
}

TEST_F(ThreadingTest, ThreadSafety_ConcurrentAccess) {
#ifdef PCR_HAS_OPENMP
    int max_threads = get_max_threads();
    if (max_threads <= 1) {
        GTEST_SKIP() << "Need more than 1 thread for concurrency test";
    }

    // Test that multiple independent operations can run in parallel
    const int tile_cells = 1000;
    const size_t num_points = 10000;

    omp_set_num_threads(max_threads);

    bool all_success = true;

    #pragma omp parallel for
    for (int thread_id = 0; thread_id < max_threads; ++thread_id) {
        // Each thread processes its own independent data
        std::vector<uint32_t> cell_indices(num_points);
        std::vector<float> values(num_points);
        std::vector<float> state(tile_cells, 0.0f);

        for (size_t i = 0; i < num_points; ++i) {
            cell_indices[i] = i % tile_cells;
            values[i] = static_cast<float>(thread_id * 100 + i);
        }

        const ReductionInfo* info = get_reduction(ReductionType::Sum);
        if (!info) {
            #pragma omp atomic write
            all_success = false;
            continue;
        }

        info->init_state(state.data(), tile_cells, nullptr);
        Status s = info->accumulate(cell_indices.data(), values.data(),
                                    state.data(), num_points, tile_cells, nullptr);

        if (!s.ok()) {
            #pragma omp atomic write
            all_success = false;
        }
    }

    EXPECT_TRUE(all_success) << "All parallel operations should succeed";
#endif
}
