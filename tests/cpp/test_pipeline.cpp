#include <gtest/gtest.h>
#include "pcr/engine/pipeline.h"
#include "pcr/core/point_cloud.h"
#include "pcr/core/grid.h"
#include <filesystem>
#include <cmath>

using namespace pcr;
namespace fs = std::filesystem;

class PipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create temp directory for test
        state_dir = "/tmp/pcr_test_pipeline";
        output_dir = "/tmp/pcr_test_output";
        fs::remove_all(state_dir);
        fs::remove_all(output_dir);
        fs::create_directories(state_dir);
        fs::create_directories(output_dir);

        // Create basic config
        config.grid.bounds = BBox{0.0, 0.0, 10.0, 10.0};
        config.grid.width = 10;
        config.grid.height = 10;
        config.grid.cell_size_x = 1.0;
        config.grid.cell_size_y = -1.0;  // North-up
        config.grid.tile_width = 5;
        config.grid.tile_height = 5;
        config.state_dir = state_dir;
    }

    void TearDown() override {
        fs::remove_all(state_dir);
        fs::remove_all(output_dir);
    }

    std::string state_dir;
    std::string output_dir;
    PipelineConfig config;
};

TEST_F(PipelineTest, Create_Validate) {
    // Add a reduction
    ReductionSpec reduction;
    reduction.value_channel = "intensity";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    Status s = pipeline->validate();
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(PipelineTest, Validate_NoReductions) {
    // No reductions specified
    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    Status s = pipeline->validate();
    EXPECT_FALSE(s.ok());
}

TEST_F(PipelineTest, SingleCloud_Sum) {
    // Create test cloud with 100 points
    auto cloud = PointCloud::create(100, MemoryLocation::Host);
    cloud->resize(100);

    // Place points in a grid pattern
    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            int idx = i * 10 + j;
            x[idx] = 0.5 + j;  // Center of cell
            y[idx] = 9.5 - i;  // Center of cell (north-up)
        }
    }

    // Add intensity channel
    cloud->add_channel("intensity", DataType::Float32);
    float* intensity = cloud->channel_f32("intensity");
    for (int i = 0; i < 100; ++i) {
        intensity[i] = 1.0f;  // Each point contributes 1
    }

    // Configure pipeline
    ReductionSpec reduction;
    reduction.value_channel = "intensity";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // Process cloud
    Status s = pipeline->ingest(*cloud);
    ASSERT_TRUE(s.ok()) << s.message;

    // Finalize
    s = pipeline->finalize();
    ASSERT_TRUE(s.ok()) << s.message;

    // Check result
    const Grid* result = pipeline->result();
    ASSERT_NE(result, nullptr);

    EXPECT_EQ(result->cols(), 10);
    EXPECT_EQ(result->rows(), 10);
    EXPECT_EQ(result->num_bands(), 1);

    // Each cell should have value 1.0 (one point per cell)
    const float* band_data = result->band_f32(0);
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(band_data[i], 1.0f) << "Cell " << i;
    }
}

TEST_F(PipelineTest, SingleCloud_Average) {
    // Create test cloud with multiple points per cell
    auto cloud = PointCloud::create(200, MemoryLocation::Host);
    cloud->resize(200);

    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());
    cloud->add_channel("value", DataType::Float32);
    float* values = cloud->channel_f32("value");

    // Put 2 points in each cell with different values
    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            int idx = (i * 10 + j) * 2;
            // First point
            x[idx] = 0.3 + j;
            y[idx] = 9.7 - i;
            values[idx] = 10.0f;

            // Second point
            x[idx + 1] = 0.7 + j;
            y[idx + 1] = 9.3 - i;
            values[idx + 1] = 20.0f;
        }
    }

    // Configure pipeline for Average
    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Average;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    Status s = pipeline->ingest(*cloud);
    ASSERT_TRUE(s.ok()) << s.message;

    s = pipeline->finalize();
    ASSERT_TRUE(s.ok()) << s.message;

    const Grid* result = pipeline->result();
    ASSERT_NE(result, nullptr);

    // Each cell should have average of 10 and 20 = 15
    const float* band_data = result->band_f32(0);
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(band_data[i], 15.0f) << "Cell " << i;
    }
}

TEST_F(PipelineTest, MultipleReductions) {
    // Create test cloud
    auto cloud = PointCloud::create(100, MemoryLocation::Host);
    cloud->resize(100);

    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());
    cloud->add_channel("intensity", DataType::Float32);
    float* intensity = cloud->channel_f32("intensity");

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            int idx = i * 10 + j;
            x[idx] = 0.5 + j;
            y[idx] = 9.5 - i;
            intensity[idx] = static_cast<float>(idx);
        }
    }

    // Configure multiple reductions
    ReductionSpec sum_reduction;
    sum_reduction.value_channel = "intensity";
    sum_reduction.type = ReductionType::Sum;
    config.reductions.push_back(sum_reduction);

    ReductionSpec max_reduction;
    max_reduction.value_channel = "intensity";
    max_reduction.type = ReductionType::Max;
    config.reductions.push_back(max_reduction);

    ReductionSpec count_reduction;
    count_reduction.value_channel = "intensity";
    count_reduction.type = ReductionType::Count;
    config.reductions.push_back(count_reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    Status s = pipeline->ingest(*cloud);
    ASSERT_TRUE(s.ok()) << s.message;

    s = pipeline->finalize();
    ASSERT_TRUE(s.ok()) << s.message;

    const Grid* result = pipeline->result();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->num_bands(), 3);

    // Band 0: Sum
    const float* sum_band = result->band_f32(0);
    // Band 1: Max
    const float* max_band = result->band_f32(1);
    // Band 2: Count
    const float* count_band = result->band_f32(2);

    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(sum_band[i], static_cast<float>(i));
        EXPECT_FLOAT_EQ(max_band[i], static_cast<float>(i));
        EXPECT_FLOAT_EQ(count_band[i], 1.0f);
    }
}

TEST_F(PipelineTest, MultipleClouds) {
    // Create two clouds
    auto cloud1 = PointCloud::create(50, MemoryLocation::Host);
    cloud1->resize(50);
    auto cloud2 = PointCloud::create(50, MemoryLocation::Host);
    cloud2->resize(50);

    // Cloud 1: first 50 cells
    double* x1 = const_cast<double*>(cloud1->x());
    double* y1 = const_cast<double*>(cloud1->y());
    cloud1->add_channel("value", DataType::Float32);
    float* val1 = cloud1->channel_f32("value");

    for (int i = 0; i < 50; ++i) {
        int row = i / 10;
        int col = i % 10;
        x1[i] = 0.5 + col;
        y1[i] = 9.5 - row;
        val1[i] = 10.0f;
    }

    // Cloud 2: overlapping cells with different values
    double* x2 = const_cast<double*>(cloud2->x());
    double* y2 = const_cast<double*>(cloud2->y());
    cloud2->add_channel("value", DataType::Float32);
    float* val2 = cloud2->channel_f32("value");

    for (int i = 0; i < 50; ++i) {
        int row = i / 10;
        int col = i % 10;
        x2[i] = 0.5 + col;
        y2[i] = 9.5 - row;
        val2[i] = 20.0f;
    }

    // Configure pipeline for Sum
    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // Ingest both clouds
    Status s = pipeline->ingest(*cloud1);
    ASSERT_TRUE(s.ok()) << s.message;

    s = pipeline->ingest(*cloud2);
    ASSERT_TRUE(s.ok()) << s.message;

    s = pipeline->finalize();
    ASSERT_TRUE(s.ok()) << s.message;

    const Grid* result = pipeline->result();
    ASSERT_NE(result, nullptr);

    const float* band_data = result->band_f32(0);

    // First 50 cells should have sum = 30 (10 + 20)
    for (int i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(band_data[i], 30.0f) << "Cell " << i;
    }

    // Remaining cells should be NaN (no data)
    for (int i = 50; i < 100; ++i) {
        EXPECT_TRUE(std::isnan(band_data[i])) << "Cell " << i;
    }
}

TEST_F(PipelineTest, DISABLED_WithFilter) {
    // Create test cloud
    auto cloud = PointCloud::create(100, MemoryLocation::Host);
    cloud->resize(100);

    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());
    cloud->add_channel("intensity", DataType::Float32);
    cloud->add_channel("classification", DataType::Float32);

    float* intensity = cloud->channel_f32("intensity");
    float* classification = cloud->channel_f32("classification");

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
            int idx = i * 10 + j;
            x[idx] = 0.5 + j;
            y[idx] = 9.5 - i;
            intensity[idx] = 1.0f;
            classification[idx] = static_cast<float>(idx % 2);  // 0 or 1
        }
    }

    // Configure filter: only class 1
    config.filter.add("classification", CompareOp::Equal, 1.0f);

    // Configure pipeline
    ReductionSpec reduction;
    reduction.value_channel = "intensity";
    reduction.type = ReductionType::Count;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    Status s = pipeline->ingest(*cloud);
    ASSERT_TRUE(s.ok()) << s.message;

    s = pipeline->finalize();
    ASSERT_TRUE(s.ok()) << s.message;

    const Grid* result = pipeline->result();
    ASSERT_NE(result, nullptr);

    const float* band_data = result->band_f32(0);

    // Check that only half the points were counted (those with class 1)
    size_t total_count = 0;
    for (int i = 0; i < 100; ++i) {
        if (!std::isnan(band_data[i])) {
            total_count += static_cast<size_t>(band_data[i]);
        }
    }

    EXPECT_EQ(total_count, 50);  // 50 points with class 1
}

TEST_F(PipelineTest, WriteGeoTiff) {
    // Create test cloud
    auto cloud = PointCloud::create(100, MemoryLocation::Host);
    cloud->resize(100);

    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());
    cloud->add_channel("value", DataType::Float32);
    float* values = cloud->channel_f32("value");

    for (int i = 0; i < 100; ++i) {
        x[i] = 0.5 + (i % 10);
        y[i] = 9.5 - (i / 10);
        values[i] = static_cast<float>(i);
    }

    // Configure pipeline with output path
    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);
    config.output_path = output_dir + "/test_output.tif";

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    Status s = pipeline->ingest(*cloud);
    ASSERT_TRUE(s.ok()) << s.message;

    s = pipeline->finalize();
    ASSERT_TRUE(s.ok()) << s.message;

    // Check that file was created
    EXPECT_TRUE(fs::exists(config.output_path));
}

TEST_F(PipelineTest, RunConvenience) {
    // Create two clouds
    auto cloud1 = PointCloud::create(50, MemoryLocation::Host);
    cloud1->resize(50);
    auto cloud2 = PointCloud::create(50, MemoryLocation::Host);
    cloud2->resize(50);

    // Fill clouds
    for (int i = 0; i < 50; ++i) {
        const_cast<double*>(cloud1->x())[i] = 0.5 + (i % 10);
        const_cast<double*>(cloud1->y())[i] = 9.5 - (i / 10);
        const_cast<double*>(cloud2->x())[i] = 0.5 + (i % 10);
        const_cast<double*>(cloud2->y())[i] = 4.5 - (i / 10);
    }

    cloud1->add_channel("v", DataType::Float32);
    cloud2->add_channel("v", DataType::Float32);
    for (int i = 0; i < 50; ++i) {
        cloud1->channel_f32("v")[i] = 1.0f;
        cloud2->channel_f32("v")[i] = 1.0f;
    }

    // Configure pipeline
    ReductionSpec reduction;
    reduction.value_channel = "v";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // Use run() convenience method
    std::vector<const PointCloud*> clouds = {cloud1.get(), cloud2.get()};
    Status s = pipeline->run(clouds);
    ASSERT_TRUE(s.ok()) << s.message;

    const Grid* result = pipeline->result();
    ASSERT_NE(result, nullptr);

    // Check stats
    ProgressInfo stats = pipeline->stats();
    EXPECT_EQ(stats.collections_processed, 2);
    EXPECT_EQ(stats.points_processed, 100);
}

TEST_F(PipelineTest, ProgressCallback) {
    auto cloud = PointCloud::create(100, MemoryLocation::Host);
    cloud->resize(100);

    for (int i = 0; i < 100; ++i) {
        const_cast<double*>(cloud->x())[i] = 0.5 + (i % 10);
        const_cast<double*>(cloud->y())[i] = 9.5 - (i / 10);
    }

    cloud->add_channel("v", DataType::Float32);
    for (int i = 0; i < 100; ++i) {
        cloud->channel_f32("v")[i] = 1.0f;
    }

    ReductionSpec reduction;
    reduction.value_channel = "v";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // Set progress callback
    int callback_count = 0;
    pipeline->set_progress_callback([&callback_count](const ProgressInfo& info) {
        callback_count++;
        EXPECT_GT(info.elapsed_seconds, 0.0f);
        return true;  // Continue
    });

    Status s = pipeline->ingest(*cloud);
    ASSERT_TRUE(s.ok()) << s.message;

    s = pipeline->finalize();
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_GT(callback_count, 0);
}

TEST_F(PipelineTest, DISABLED_EmptyCloud) {
    auto cloud = PointCloud::create(0, MemoryLocation::Host);
    cloud->resize(0);

    ReductionSpec reduction;
    reduction.value_channel = "v";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // Should handle empty cloud gracefully
    Status s = pipeline->ingest(*cloud);
    EXPECT_TRUE(s.ok()) << s.message;

    s = pipeline->finalize();
    EXPECT_TRUE(s.ok()) << s.message;
}
