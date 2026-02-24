#include "pcr/core/point_cloud.h"
#include "test_helpers.h"
#include <gtest/gtest.h>

using namespace pcr;
using namespace pcr::test;

// ===========================================================================
// Construction Tests
// ===========================================================================

TEST(PointCloudTest, CreateValid) {
    auto cloud = PointCloud::create(100);

    ASSERT_NE(cloud, nullptr);
    EXPECT_EQ(cloud->capacity(), 100);
    EXPECT_EQ(cloud->count(), 0);
    EXPECT_EQ(cloud->location(), MemoryLocation::Host);
    EXPECT_NE(cloud->x(), nullptr);
    EXPECT_NE(cloud->y(), nullptr);
}

TEST(PointCloudTest, CreateZeroCapacity) {
    auto cloud = PointCloud::create(0);
    EXPECT_EQ(cloud, nullptr);
}

TEST(PointCloudTest, WrapExternalBuffers) {
    double x_data[10];
    double y_data[10];

    for (int i = 0; i < 10; ++i) {
        x_data[i] = static_cast<double>(i);
        y_data[i] = static_cast<double>(i) * 2.0;
    }

    auto cloud = PointCloud::wrap(x_data, y_data, 10);

    ASSERT_NE(cloud, nullptr);
    EXPECT_EQ(cloud->capacity(), 10);
    EXPECT_EQ(cloud->count(), 10);
    EXPECT_EQ(cloud->x(), x_data);
    EXPECT_EQ(cloud->y(), y_data);
}

TEST(PointCloudTest, WrapNullPointers) {
    auto cloud1 = PointCloud::wrap(nullptr, nullptr, 10);
    EXPECT_EQ(cloud1, nullptr);

    double data[10];
    auto cloud2 = PointCloud::wrap(data, nullptr, 10);
    EXPECT_EQ(cloud2, nullptr);

    auto cloud3 = PointCloud::wrap(nullptr, data, 10);
    EXPECT_EQ(cloud3, nullptr);
}

// ===========================================================================
// Channel Management Tests
// ===========================================================================

TEST(PointCloudTest, AddChannelFloat32) {
    auto cloud = make_test_point_cloud(100);

    Status s = cloud->add_channel("intensity", DataType::Float32);
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(cloud->has_channel("intensity"));
}

TEST(PointCloudTest, AddChannelInt32) {
    auto cloud = make_test_point_cloud(100);

    Status s = cloud->add_channel("classification", DataType::Int32);
    EXPECT_TRUE(s.ok());
    EXPECT_TRUE(cloud->has_channel("classification"));
}

TEST(PointCloudTest, AddMultipleChannels) {
    auto cloud = make_test_point_cloud(100);

    cloud->add_channel("intensity", DataType::Float32);
    cloud->add_channel("red", DataType::UInt8);
    cloud->add_channel("green", DataType::UInt8);
    cloud->add_channel("blue", DataType::UInt8);

    EXPECT_TRUE(cloud->has_channel("intensity"));
    EXPECT_TRUE(cloud->has_channel("red"));
    EXPECT_TRUE(cloud->has_channel("green"));
    EXPECT_TRUE(cloud->has_channel("blue"));
    EXPECT_FALSE(cloud->has_channel("nonexistent"));
}

TEST(PointCloudTest, DuplicateChannelName) {
    auto cloud = make_test_point_cloud(100);

    Status s1 = cloud->add_channel("test", DataType::Float32);
    EXPECT_TRUE(s1.ok());

    Status s2 = cloud->add_channel("test", DataType::Float32);
    EXPECT_FALSE(s2.ok());
    EXPECT_EQ(s2.code, StatusCode::InvalidArgument);
}

TEST(PointCloudTest, ChannelDescriptor) {
    auto cloud = make_test_point_cloud(100);
    cloud->add_channel("intensity", DataType::Float32);

    const ChannelDesc* desc = cloud->channel("intensity");
    ASSERT_NE(desc, nullptr);
    EXPECT_EQ(desc->name, "intensity");
    EXPECT_EQ(desc->dtype, DataType::Float32);

    const ChannelDesc* desc_missing = cloud->channel("nonexistent");
    EXPECT_EQ(desc_missing, nullptr);
}

TEST(PointCloudTest, ChannelNames) {
    auto cloud = make_test_point_cloud(100);
    cloud->add_channel("a", DataType::Float32);
    cloud->add_channel("b", DataType::Int32);
    cloud->add_channel("c", DataType::UInt8);

    auto names = cloud->channel_names();
    EXPECT_EQ(names.size(), 3);

    // Names might be in any order (hash map)
    EXPECT_NE(std::find(names.begin(), names.end(), "a"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "b"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "c"), names.end());
}

// ===========================================================================
// Data Access Tests
// ===========================================================================

TEST(PointCloudTest, CoordinateAccess) {
    auto cloud = PointCloud::create(10);
    fill_test_points(*cloud, 10);

    const double* x = cloud->x();
    const double* y = cloud->y();

    for (int i = 0; i < 10; ++i) {
        EXPECT_DOUBLE_EQ(x[i], static_cast<double>(i));
        EXPECT_DOUBLE_EQ(y[i], static_cast<double>(i) * 2.0);
    }
}

TEST(PointCloudTest, ChannelDataRawAccess) {
    auto cloud = make_test_point_cloud(100);
    cloud->add_channel("intensity", DataType::Float32);

    void* data = cloud->channel_data("intensity");
    EXPECT_NE(data, nullptr);

    const void* data_const = const_cast<const PointCloud*>(cloud.get())->channel_data("intensity");
    EXPECT_NE(data_const, nullptr);
    EXPECT_EQ(data, data_const);

    void* data_missing = cloud->channel_data("nonexistent");
    EXPECT_EQ(data_missing, nullptr);
}

TEST(PointCloudTest, ChannelF32TypedAccess) {
    auto cloud = make_test_point_cloud(100);
    cloud->add_channel("intensity", DataType::Float32);

    float* data = cloud->channel_f32("intensity");
    ASSERT_NE(data, nullptr);

    // Write some test data
    for (int i = 0; i < 100; ++i) {
        data[i] = static_cast<float>(i) * 0.5f;
    }

    // Read back
    const float* data_const = const_cast<const PointCloud*>(cloud.get())->channel_f32("intensity");
    EXPECT_FLOAT_EQ(data_const[50], 25.0f);
}

TEST(PointCloudTest, ChannelI32TypedAccess) {
    auto cloud = make_test_point_cloud(100);
    cloud->add_channel("classification", DataType::Int32);

    int32_t* data = cloud->channel_i32("classification");
    ASSERT_NE(data, nullptr);

    data[0] = 42;
    data[1] = -10;

    const int32_t* data_const = const_cast<const PointCloud*>(cloud.get())->channel_i32("classification");
    EXPECT_EQ(data_const[0], 42);
    EXPECT_EQ(data_const[1], -10);
}

TEST(PointCloudTest, ChannelTypeMismatch) {
    auto cloud = make_test_point_cloud(100);
    cloud->add_channel("intensity", DataType::Float32);

    // Try to access Float32 channel as Int32
    int32_t* data = cloud->channel_i32("intensity");
    EXPECT_EQ(data, nullptr);
}

// ===========================================================================
// Resize Tests
// ===========================================================================

TEST(PointCloudTest, ResizeWithinCapacity) {
    auto cloud = make_test_point_cloud(100);
    EXPECT_EQ(cloud->count(), 0);

    Status s = cloud->resize(50);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(cloud->count(), 50);
    EXPECT_EQ(cloud->capacity(), 100);
}

TEST(PointCloudTest, ResizeBeyondCapacity) {
    auto cloud = make_test_point_cloud(100);

    Status s = cloud->resize(200);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}

TEST(PointCloudTest, ResizeToZero) {
    auto cloud = make_test_point_cloud(100);
    cloud->resize(50);

    Status s = cloud->resize(0);
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(cloud->count(), 0);
}

// ===========================================================================
// CRS Tests
// ===========================================================================

TEST(PointCloudTest, CRSRoundTrip) {
    auto cloud = make_test_point_cloud(100);

    CRS crs_in = CRS::from_epsg(4326);
    cloud->set_crs(crs_in);

    CRS crs_out = cloud->crs();
    EXPECT_EQ(crs_out.epsg, 4326);
    EXPECT_TRUE(crs_out.equivalent_to(crs_in));
}

TEST(PointCloudTest, DefaultCRS) {
    auto cloud = make_test_point_cloud(100);

    CRS crs = cloud->crs();
    EXPECT_FALSE(crs.is_valid());
}

// ===========================================================================
// Memory Operations Tests
// ===========================================================================

TEST(PointCloudTest, CopyToSameLocation) {
    auto cloud1 = make_test_point_cloud(100);
    fill_test_points(*cloud1, 50);
    cloud1->add_channel("intensity", DataType::Float32);

    float* intensity1 = cloud1->channel_f32("intensity");
    for (int i = 0; i < 50; ++i) {
        intensity1[i] = static_cast<float>(i);
    }

    CRS crs = CRS::from_epsg(3857);
    cloud1->set_crs(crs);

    auto cloud2 = cloud1->to(MemoryLocation::Host);

    ASSERT_NE(cloud2, nullptr);
    EXPECT_EQ(cloud2->count(), 50);
    EXPECT_EQ(cloud2->capacity(), 100);

    // Verify coordinates
    const double* x2 = cloud2->x();
    const double* y2 = cloud2->y();
    EXPECT_DOUBLE_EQ(x2[0], 0.0);
    EXPECT_DOUBLE_EQ(y2[0], 0.0);
    EXPECT_DOUBLE_EQ(x2[49], 49.0);

    // Verify channel data
    const float* intensity2 = cloud2->channel_f32("intensity");
    ASSERT_NE(intensity2, nullptr);
    EXPECT_FLOAT_EQ(intensity2[0], 0.0f);
    EXPECT_FLOAT_EQ(intensity2[49], 49.0f);

    // Verify CRS
    CRS crs2 = cloud2->crs();
    EXPECT_TRUE(crs2.equivalent_to(crs));
}

// ===========================================================================
// Wrapped vs Owned Tests
// ===========================================================================

TEST(PointCloudTest, WrappedDataModification) {
    double x_data[10];
    double y_data[10];

    for (int i = 0; i < 10; ++i) {
        x_data[i] = static_cast<double>(i);
        y_data[i] = static_cast<double>(i);
    }

    auto cloud = PointCloud::wrap(x_data, y_data, 10);

    // Modify external buffer
    x_data[5] = 999.0;

    // Cloud should see the change
    EXPECT_DOUBLE_EQ(cloud->x()[5], 999.0);
}

TEST(PointCloudTest, OwnedDataIndependent) {
    auto cloud = PointCloud::create(10);
    fill_test_points(*cloud, 10);

    double* x = cloud->x();
    double original_x0 = x[0];

    x[0] = 999.0;

    // Verify modification persists
    EXPECT_DOUBLE_EQ(cloud->x()[0], 999.0);
    EXPECT_NE(original_x0, 999.0);
}

// ===========================================================================
// Edge Cases Tests
// ===========================================================================

TEST(PointCloudTest, ZeroPoints) {
    auto cloud = make_test_point_cloud(100);
    EXPECT_EQ(cloud->count(), 0);

    // Should be safe to access arrays (just don't dereference)
    EXPECT_NE(cloud->x(), nullptr);
    EXPECT_NE(cloud->y(), nullptr);
}

TEST(PointCloudTest, ManyChannels) {
    auto cloud = make_test_point_cloud(100);

    for (int i = 0; i < 20; ++i) {
        std::string name = "channel" + std::to_string(i);
        Status s = cloud->add_channel(name, DataType::Float32);
        EXPECT_TRUE(s.ok());
    }

    EXPECT_EQ(cloud->channel_names().size(), 20);
}

TEST(PointCloudTest, LargeCloud) {
    // 1M points
    auto cloud = PointCloud::create(1000000);

    ASSERT_NE(cloud, nullptr);
    EXPECT_EQ(cloud->capacity(), 1000000);

    Status s = cloud->resize(1000000);
    EXPECT_TRUE(s.ok());

    // Spot check
    double* x = cloud->x();
    x[0] = 0.0;
    x[999999] = 999999.0;
    EXPECT_DOUBLE_EQ(x[0], 0.0);
    EXPECT_DOUBLE_EQ(x[999999], 999999.0);
}
