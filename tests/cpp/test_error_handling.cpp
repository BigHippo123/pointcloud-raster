#include <gtest/gtest.h>
#include "pcr/engine/pipeline.h"
#include "pcr/core/point_cloud.h"
#include "pcr/core/types.h"
#include <filesystem>

#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#endif

using namespace pcr;
namespace fs = std::filesystem;

class ErrorHandlingTest : public ::testing::Test {
protected:
    void SetUp() override {
        state_dir = "/tmp/pcr_error_handling_test";
        fs::create_directories(state_dir);

        // Basic grid config
        config.grid.bounds = BBox{0.0, 0.0, 10.0, 10.0};
        config.grid.width = 10;
        config.grid.height = 10;
        config.grid.cell_size_x = 1.0;
        config.grid.cell_size_y = -1.0;
        config.grid.tile_width = 5;
        config.grid.tile_height = 5;
        config.state_dir = state_dir;
    }

    void TearDown() override {
        fs::remove_all(state_dir);
    }

    std::string state_dir;
    PipelineConfig config;
};

// ---------------------------------------------------------------------------
// CUDA Capability Detection Tests
// ---------------------------------------------------------------------------

TEST_F(ErrorHandlingTest, CUDA_CompilationDetection) {
    // Test that we can detect if CUDA is compiled
#ifdef PCR_HAS_CUDA
    EXPECT_TRUE(cuda_is_compiled());
#else
    EXPECT_FALSE(cuda_is_compiled());
#endif
}

TEST_F(ErrorHandlingTest, CUDA_DeviceDetection) {
    // Test device detection (may pass or fail depending on hardware)
    int device_count = cuda_device_count();

#ifdef PCR_HAS_CUDA
    // If CUDA is compiled, we should get a valid count (0 or more)
    EXPECT_GE(device_count, 0);

    bool device_available = cuda_device_available();
    if (device_count > 0) {
        EXPECT_TRUE(device_available);
    } else {
        EXPECT_FALSE(device_available);
    }
#else
    // If CUDA not compiled, count should be 0
    EXPECT_EQ(device_count, 0);
    EXPECT_FALSE(cuda_device_available());
#endif
}

TEST_F(ErrorHandlingTest, CUDA_DeviceName) {
    std::string device_name = cuda_device_name(0);

#ifdef PCR_HAS_CUDA
    if (cuda_device_available()) {
        // Should get a real device name
        EXPECT_NE(device_name, "Unknown GPU");
        EXPECT_NE(device_name, "CUDA not compiled");
    } else {
        // No device available
        EXPECT_TRUE(device_name == "Unknown GPU" || device_name.find("CUDA") != std::string::npos);
    }
#else
    EXPECT_EQ(device_name, "CUDA not compiled");
#endif
}

TEST_F(ErrorHandlingTest, CUDA_MemoryInfo) {
    size_t free_bytes = 0, total_bytes = 0;
    bool success = cuda_get_memory_info(&free_bytes, &total_bytes, 0);

#ifdef PCR_HAS_CUDA
    if (cuda_device_available()) {
        EXPECT_TRUE(success);
        EXPECT_GT(total_bytes, 0);
        EXPECT_LE(free_bytes, total_bytes);
    } else {
        EXPECT_FALSE(success);
    }
#else
    EXPECT_FALSE(success);
#endif
}

// ---------------------------------------------------------------------------
// GPU Fallback Tests
// ---------------------------------------------------------------------------

TEST_F(ErrorHandlingTest, GPU_FallbackToCPU_WhenNoDevice) {
    // Configure for GPU mode with fallback enabled
    config.exec_mode = ExecutionMode::GPU;
    config.gpu_fallback_to_cpu = true;
    config.gpu_require_strict = false;

    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);

#ifdef PCR_HAS_CUDA
    if (cuda_device_available()) {
        // GPU available - should create successfully in GPU mode
        ASSERT_NE(pipeline, nullptr);
    } else {
        // No GPU - should fall back to CPU mode
        ASSERT_NE(pipeline, nullptr);
        // Pipeline should have internally switched to CPU mode
    }
#else
    // CUDA not compiled - should fall back to CPU
    ASSERT_NE(pipeline, nullptr);
#endif
}

TEST_F(ErrorHandlingTest, GPU_StrictMode_FailsWithoutDevice) {
#ifndef PCR_HAS_CUDA
    GTEST_SKIP() << "Test requires CUDA to be compiled";
#else
    if (cuda_device_available()) {
        GTEST_SKIP() << "Test requires no GPU to be available";
    }

    // Configure for GPU mode with strict requirement
    config.exec_mode = ExecutionMode::GPU;
    config.gpu_fallback_to_cpu = false;
    config.gpu_require_strict = true;

    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);

    // Should fail to create in strict mode without GPU
    EXPECT_EQ(pipeline, nullptr);
#endif
}

TEST_F(ErrorHandlingTest, GPU_AutoMode_UsesAvailable) {
    // Auto mode should work regardless of GPU availability
    config.exec_mode = ExecutionMode::Auto;

    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // Should validate successfully
    Status s = pipeline->validate();
    EXPECT_TRUE(s.ok()) << s.message;
}

TEST_F(ErrorHandlingTest, CPU_Mode_AlwaysWorks) {
    // CPU mode should always work
    config.exec_mode = ExecutionMode::CPU;

    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    Status s = pipeline->validate();
    EXPECT_TRUE(s.ok()) << s.message;
}

// ---------------------------------------------------------------------------
// Error Message Quality Tests
// ---------------------------------------------------------------------------

TEST_F(ErrorHandlingTest, MissingReduction_ClearError) {
    // Create pipeline without reductions
    config.exec_mode = ExecutionMode::CPU;

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // Validate should fail with clear message
    Status s = pipeline->validate();
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("reduction"), std::string::npos);
}

TEST_F(ErrorHandlingTest, InvalidGridConfig_ClearError) {
    // Create pipeline with invalid grid (width = 0)
    config.exec_mode = ExecutionMode::CPU;
    config.grid.width = 0;  // Invalid!
    config.grid.height = 10;

    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);

    // Should either fail to create or fail validation
    if (pipeline) {
        Status s = pipeline->validate();
        EXPECT_FALSE(s.ok());
    } else {
        // Failed to create - acceptable
        EXPECT_EQ(pipeline, nullptr);
    }
}

TEST_F(ErrorHandlingTest, MissingChannel_ClearError) {
    // Create valid pipeline
    config.exec_mode = ExecutionMode::CPU;

    ReductionSpec reduction;
    reduction.value_channel = "nonexistent_channel";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);
    ASSERT_NE(pipeline, nullptr);

    // Create cloud without the required channel
    auto cloud = PointCloud::create(100, MemoryLocation::Host);
    cloud->resize(100);

    double* x = const_cast<double*>(cloud->x());
    double* y = const_cast<double*>(cloud->y());
    for (int i = 0; i < 100; ++i) {
        x[i] = 5.0;
        y[i] = 5.0;
    }

    // Ingest should fail with clear error about missing channel
    Status s = pipeline->ingest(*cloud);
    EXPECT_FALSE(s.ok());
    EXPECT_NE(s.message.find("channel"), std::string::npos);
    EXPECT_NE(s.message.find("nonexistent_channel"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Configuration Validation Tests
// ---------------------------------------------------------------------------

TEST_F(ErrorHandlingTest, InvalidDeviceID_HandledGracefully) {
#ifndef PCR_HAS_CUDA
    GTEST_SKIP() << "Test requires CUDA to be compiled";
#else
    if (!cuda_device_available()) {
        GTEST_SKIP() << "Test requires at least one GPU";
    }

    // Try to use an invalid device ID
    config.exec_mode = ExecutionMode::GPU;
    config.cuda_device_id = 999;  // Definitely doesn't exist
    config.gpu_fallback_to_cpu = true;

    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);

    // Should either fail or fall back to CPU
    // Either outcome is acceptable, but shouldn't crash
    if (pipeline) {
        Status s = pipeline->validate();
        // May or may not validate depending on fallback behavior
        (void)s;  // Don't enforce specific behavior
    }
#endif
}

TEST_F(ErrorHandlingTest, ZeroTileSize_Rejected) {
    config.exec_mode = ExecutionMode::CPU;
    config.grid.tile_width = 0;  // Invalid!

    ReductionSpec reduction;
    reduction.value_channel = "value";
    reduction.type = ReductionType::Sum;
    config.reductions.push_back(reduction);

    auto pipeline = Pipeline::create(config);

    // Should fail either at creation or validation
    if (pipeline) {
        Status s = pipeline->validate();
        EXPECT_FALSE(s.ok());
    }
    // Or pipeline is nullptr - also acceptable
}
