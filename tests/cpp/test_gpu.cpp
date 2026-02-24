#include <gtest/gtest.h>
#include "pcr/core/point_cloud.h"
#include "pcr/engine/pipeline.h"
#include "pcr/engine/tile_manager.h"
#include "pcr/core/grid.h"
#include "pcr/core/grid_config.h"
#include <cmath>

#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace pcr;

// Helper to check if CUDA is available
bool cuda_available() {
#ifdef PCR_HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

// Test fixture for GPU tests
class GPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!cuda_available()) {
            GTEST_SKIP() << "CUDA not available, skipping GPU tests";
        }
    }
};

// ---------------------------------------------------------------------------
// PointCloud GPU Memory Allocation Tests
// ---------------------------------------------------------------------------

TEST_F(GPUTest, PointCloud_CreateOnDevice) {
#ifdef PCR_HAS_CUDA
    // Create point cloud on device
    auto cloud = PointCloud::create(100, MemoryLocation::Device);
    ASSERT_NE(cloud, nullptr);
    EXPECT_EQ(cloud->capacity(), 100);
    EXPECT_EQ(cloud->location(), MemoryLocation::Device);

    // Coordinates should be allocated
    EXPECT_NE(cloud->x(), nullptr);
    EXPECT_NE(cloud->y(), nullptr);
#endif
}

TEST_F(GPUTest, PointCloud_CreateOnHostPinned) {
#ifdef PCR_HAS_CUDA
    // Create point cloud on host pinned memory
    auto cloud = PointCloud::create(100, MemoryLocation::HostPinned);
    ASSERT_NE(cloud, nullptr);
    EXPECT_EQ(cloud->capacity(), 100);
    EXPECT_EQ(cloud->location(), MemoryLocation::HostPinned);

    EXPECT_NE(cloud->x(), nullptr);
    EXPECT_NE(cloud->y(), nullptr);
#endif
}

TEST_F(GPUTest, PointCloud_AddChannelOnDevice) {
#ifdef PCR_HAS_CUDA
    auto cloud = PointCloud::create(50, MemoryLocation::Device);
    ASSERT_NE(cloud, nullptr);

    // Add a channel
    Status s = cloud->add_channel("intensity", DataType::Float32);
    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_TRUE(cloud->has_channel("intensity"));

    // Channel data should be allocated
    EXPECT_NE(cloud->channel_data("intensity"), nullptr);
#endif
}

TEST_F(GPUTest, PointCloud_HostToDeviceTransfer) {
#ifdef PCR_HAS_CUDA
    // Create host cloud with data
    auto host_cloud = PointCloud::create(10, MemoryLocation::Host);
    ASSERT_NE(host_cloud, nullptr);

    // Fill with test data
    double* x = const_cast<double*>(host_cloud->x());
    double* y = const_cast<double*>(host_cloud->y());
    for (size_t i = 0; i < 10; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i * 2);
    }
    host_cloud->resize(10);

    // Add a channel
    host_cloud->add_channel("values", DataType::Float32);
    float* values = host_cloud->channel_f32("values");
    for (size_t i = 0; i < 10; ++i) {
        values[i] = static_cast<float>(i * 3);
    }

    // Transfer to device
    auto device_cloud = host_cloud->to(MemoryLocation::Device);
    ASSERT_NE(device_cloud, nullptr);
    EXPECT_EQ(device_cloud->location(), MemoryLocation::Device);
    EXPECT_EQ(device_cloud->count(), 10);
    EXPECT_TRUE(device_cloud->has_channel("values"));

    // Transfer back to verify
    auto host_cloud2 = device_cloud->to(MemoryLocation::Host);
    ASSERT_NE(host_cloud2, nullptr);

    const double* x2 = host_cloud2->x();
    const double* y2 = host_cloud2->y();
    const float* values2 = host_cloud2->channel_f32("values");

    for (size_t i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(x2[i], static_cast<double>(i));
        EXPECT_DOUBLE_EQ(y2[i], static_cast<double>(i * 2));
        EXPECT_FLOAT_EQ(values2[i], static_cast<float>(i * 3));
    }
#endif
}

TEST_F(GPUTest, PointCloud_AsyncDeviceTransfer) {
#ifdef PCR_HAS_CUDA
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create host cloud
    auto host_cloud = PointCloud::create(20, MemoryLocation::Host);
    ASSERT_NE(host_cloud, nullptr);

    double* x = const_cast<double*>(host_cloud->x());
    for (size_t i = 0; i < 20; ++i) {
        x[i] = static_cast<double>(i * 1.5);
    }
    host_cloud->resize(20);

    // Async transfer
    auto device_cloud = host_cloud->to_device_async(stream);
    ASSERT_NE(device_cloud, nullptr);
    EXPECT_EQ(device_cloud->location(), MemoryLocation::Device);

    // Synchronize and verify
    cudaStreamSynchronize(stream);

    auto host_verify = device_cloud->to(MemoryLocation::Host);
    const double* x_verify = host_verify->x();
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_DOUBLE_EQ(x_verify[i], static_cast<double>(i * 1.5));
    }

    cudaStreamDestroy(stream);
#endif
}

// ---------------------------------------------------------------------------
// TileManager GPU Tests
// ---------------------------------------------------------------------------

TEST_F(GPUTest, TileManager_DeviceState) {
#ifdef PCR_HAS_CUDA
    GridConfig grid_config;
    grid_config.bounds = BBox{0.0, 0.0, 10.0, 10.0};
    grid_config.cell_size_x = 1.0;
    grid_config.cell_size_y = -1.0;
    grid_config.tile_width = 10;
    grid_config.tile_height = 10;
    grid_config.compute_dimensions();

    TileManagerConfig tm_config;
    tm_config.state_dir = "/tmp/pcr_gpu_test";
    tm_config.cache_size_bytes = 10 * 1024 * 1024;  // 10MB
    tm_config.state_floats = 1;  // Sum reduction
    tm_config.grid_config = grid_config;
    tm_config.memory_location = MemoryLocation::Device;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    tm_config.cuda_stream = stream;

    auto tile_mgr = TileManager::create(tm_config);
    ASSERT_NE(tile_mgr, nullptr);

    // Acquire tile state on device
    TileIndex tile{0, 0};
    float* state_ptr = nullptr;
    Status s = tile_mgr->acquire(tile, ReductionType::Sum, &state_ptr);
    ASSERT_TRUE(s.ok()) << s.message;
    ASSERT_NE(state_ptr, nullptr);

    // The pointer should be a device pointer
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, state_ptr);
    ASSERT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);

    // Release
    s = tile_mgr->release(tile);
    EXPECT_TRUE(s.ok()) << s.message;

    cudaStreamDestroy(stream);
#endif
}

TEST_F(GPUTest, TileManager_HostDeviceRoundtrip) {
#ifdef PCR_HAS_CUDA
    GridConfig grid_config;
    grid_config.bounds = BBox{0.0, 0.0, 10.0, 10.0};
    grid_config.cell_size_x = 1.0;
    grid_config.cell_size_y = -1.0;
    grid_config.tile_width = 10;
    grid_config.tile_height = 10;
    grid_config.compute_dimensions();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Create tile manager with Device memory
    TileManagerConfig tm_config;
    tm_config.state_dir = "/tmp/pcr_gpu_test_roundtrip";
    tm_config.cache_size_bytes = 10 * 1024 * 1024;
    tm_config.state_floats = 2;  // Average (sum + count)
    tm_config.grid_config = grid_config;
    tm_config.memory_location = MemoryLocation::Device;
    tm_config.cuda_stream = stream;

    auto tile_mgr = TileManager::create(tm_config);
    ASSERT_NE(tile_mgr, nullptr);

    TileIndex tile{0, 0};
    float* device_state = nullptr;

    // Acquire (should initialize on device)
    Status s = tile_mgr->acquire(tile, ReductionType::Average, &device_state);
    ASSERT_TRUE(s.ok()) << s.message;

    // Modify state on device (set to known values)
    int tile_cells = 100;  // 10x10
    std::vector<float> host_values(tile_cells * 2);
    for (int i = 0; i < tile_cells; ++i) {
        host_values[i] = 42.0f;                    // sum
        host_values[tile_cells + i] = 7.0f;        // count
    }

    cudaMemcpy(device_state, host_values.data(),
               host_values.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Release (should transfer back to host)
    s = tile_mgr->release(tile);
    ASSERT_TRUE(s.ok()) << s.message;

    // Flush to disk
    s = tile_mgr->flush_all();
    ASSERT_TRUE(s.ok()) << s.message;

    // Clear cache to force reload
    tile_mgr->clear_cache();

    // Acquire again (should load from disk and transfer to device)
    s = tile_mgr->acquire(tile, ReductionType::Average, &device_state);
    ASSERT_TRUE(s.ok()) << s.message;

    // Transfer back and verify values
    std::vector<float> verify_values(tile_cells * 2);
    cudaMemcpy(verify_values.data(), device_state,
               verify_values.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < tile_cells; ++i) {
        EXPECT_FLOAT_EQ(verify_values[i], 42.0f) << "at index " << i;
        EXPECT_FLOAT_EQ(verify_values[tile_cells + i], 7.0f) << "at index " << (tile_cells + i);
    }

    tile_mgr->release(tile);
    cudaStreamDestroy(stream);
#endif
}

// ---------------------------------------------------------------------------
// Pipeline GPU Tests
// ---------------------------------------------------------------------------

TEST_F(GPUTest, Pipeline_CPUModeBaseline) {
#ifdef PCR_HAS_CUDA
    // Test that CPU mode still works (baseline for comparison)
    GridConfig grid_config;
    grid_config.bounds = BBox{0.0, 0.0, 100.0, 100.0};
    grid_config.cell_size_x = 1.0;
    grid_config.cell_size_y = -1.0;
    grid_config.tile_width = 100;
    grid_config.tile_height = 100;
    grid_config.compute_dimensions();

    // Create pipeline with CPU mode
    PipelineConfig config;
    config.grid = grid_config;
    config.exec_mode = ExecutionMode::CPU;
    config.state_dir = "/tmp/pcr_cpu_baseline_test";

    ReductionSpec reduction;
    reduction.value_channel = "intensity";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    Status s = pipeline->validate();
    ASSERT_TRUE(s.ok()) << s.message;

    // Create test point cloud on host
    auto cloud = PointCloud::create(100, MemoryLocation::Host);
    ASSERT_NE(cloud, nullptr);

    // Fill with test data
    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());
    for (size_t i = 0; i < 100; ++i) {
        x[i] = 50.0 + (i % 10);
        y[i] = 50.0 + (i / 10);
    }
    cloud->resize(100);

    s = cloud->add_channel("intensity", DataType::Float32);
    ASSERT_TRUE(s.ok()) << s.message;

    float* intensity = cloud->channel_f32("intensity");
    for (size_t i = 0; i < 100; ++i) {
        intensity[i] = 1.0f;
    }

    // Process with CPU mode
    s = pipeline->ingest(*cloud);
    EXPECT_TRUE(s.ok()) << s.message;

    s = pipeline->finalize();
    EXPECT_TRUE(s.ok()) << s.message;

    // Verify result
    const Grid* result = pipeline->result();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result->cols(), 100);
    EXPECT_EQ(result->rows(), 100);
#endif
}

TEST_F(GPUTest, Pipeline_ConfigurationCheck) {
#ifdef PCR_HAS_CUDA
    // Just test that pipeline can be created with GPU mode
    GridConfig grid_config;
    grid_config.bounds = BBox{0.0, 0.0, 50.0, 50.0};
    grid_config.cell_size_x = 0.5;
    grid_config.cell_size_y = -0.5;
    grid_config.tile_width = 50;
    grid_config.tile_height = 50;
    grid_config.compute_dimensions();

    PipelineConfig config;
    config.grid = grid_config;
    config.exec_mode = ExecutionMode::GPU;
    config.state_dir = "/tmp/pcr_gpu_config_test";

    ReductionSpec reduction;
    reduction.value_channel = "values";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    // Should be able to create pipeline with GPU mode
    auto pipeline = Pipeline::create(config);
    EXPECT_NE(pipeline, nullptr);

    if (pipeline) {
        Status s = pipeline->validate();
        EXPECT_TRUE(s.ok()) << s.message;
    }
#endif
}

TEST_F(GPUTest, MemoryLocation_Verification) {
#ifdef PCR_HAS_CUDA
    // Create clouds in different memory locations
    auto host_cloud = PointCloud::create(10, MemoryLocation::Host);
    auto pinned_cloud = PointCloud::create(10, MemoryLocation::HostPinned);
    auto device_cloud = PointCloud::create(10, MemoryLocation::Device);

    ASSERT_NE(host_cloud, nullptr);
    ASSERT_NE(pinned_cloud, nullptr);
    ASSERT_NE(device_cloud, nullptr);

    EXPECT_EQ(host_cloud->location(), MemoryLocation::Host);
    EXPECT_EQ(pinned_cloud->location(), MemoryLocation::HostPinned);
    EXPECT_EQ(device_cloud->location(), MemoryLocation::Device);

    // Verify pointer types
    cudaPointerAttributes attrs;

    // Host memory
    cudaError_t err = cudaPointerGetAttributes(&attrs, host_cloud->x());
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeUnregistered);

    // Pinned memory
    err = cudaPointerGetAttributes(&attrs, pinned_cloud->x());
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeHost);

    // Device memory
    err = cudaPointerGetAttributes(&attrs, device_cloud->x());
    EXPECT_EQ(err, cudaSuccess);
    EXPECT_EQ(attrs.type, cudaMemoryTypeDevice);
#endif
}

// ---------------------------------------------------------------------------
// GPU Performance Sanity Check
// ---------------------------------------------------------------------------

TEST_F(GPUTest, LargeDataTransfer) {
#ifdef PCR_HAS_CUDA
    const size_t num_points = 1000000;  // 1M points

    // Create large host cloud
    auto host_cloud = PointCloud::create(num_points, MemoryLocation::Host);
    ASSERT_NE(host_cloud, nullptr);

    // Fill with data
    double* x = const_cast<double*>(host_cloud->x());
    double* y = const_cast<double*>(host_cloud->y());
    for (size_t i = 0; i < num_points; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i * 2);
    }
    host_cloud->resize(num_points);

    host_cloud->add_channel("data", DataType::Float32);
    float* data = host_cloud->channel_f32("data");
    for (size_t i = 0; i < num_points; ++i) {
        data[i] = static_cast<float>(i % 1000);
    }

    // Transfer to device
    auto device_cloud = host_cloud->to(MemoryLocation::Device);
    ASSERT_NE(device_cloud, nullptr);
    EXPECT_EQ(device_cloud->count(), num_points);

    // Transfer back
    auto host_cloud2 = device_cloud->to(MemoryLocation::Host);
    ASSERT_NE(host_cloud2, nullptr);

    // Spot check values
    const double* x2 = host_cloud2->x();
    const double* y2 = host_cloud2->y();
    const float* data2 = host_cloud2->channel_f32("data");

    EXPECT_DOUBLE_EQ(x2[0], 0.0);
    EXPECT_DOUBLE_EQ(x2[num_points - 1], static_cast<double>(num_points - 1));
    EXPECT_DOUBLE_EQ(y2[0], 0.0);
    EXPECT_FLOAT_EQ(data2[500000], static_cast<float>(500000 % 1000));
#endif
}
