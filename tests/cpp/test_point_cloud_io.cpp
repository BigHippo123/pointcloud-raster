#include "pcr/io/point_cloud_io.h"
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

class PointCloudIOTest : public ::testing::Test {
protected:
    void SetUp() override {
        test_dir_ = "/tmp/pcr_test_point_cloud_io/";
        int result = system(("mkdir -p " + test_dir_).c_str());
        (void)result;  // Suppress unused warning
    }

    void TearDown() override {
        int result = system(("rm -rf " + test_dir_).c_str());
        (void)result;
    }

    std::string test_dir_;
};

// ===========================================================================
// PCR Binary Format Tests
// ===========================================================================

TEST_F(PointCloudIOTest, PCRBinary_WriteReadRoundTrip) {
    std::string path = test_dir_ + "test.pcrp";

    // Create test point cloud
    auto cloud_out = PointCloud::create(100, MemoryLocation::Host);
    cloud_out->resize(100);  // Set count to 100
    cloud_out->set_crs(CRS::from_epsg(3857));

    // Fill with test data
    double* x = const_cast<double*>(cloud_out->x());
    double* y = const_cast<double*>(cloud_out->y());
    for (size_t i = 0; i < 100; ++i) {
        x[i] = static_cast<double>(i) * 10.0;
        y[i] = static_cast<double>(i) * 20.0;
    }

    // Add channel
    cloud_out->add_channel("intensity", DataType::Float32);
    float* intensity_out = cloud_out->channel_f32("intensity");
    for (size_t i = 0; i < 100; ++i) {
        intensity_out[i] = static_cast<float>(i) * 0.5f;
    }

    // Write
    Status s = write_point_cloud(path, *cloud_out, PointCloudFormat::PCR_Binary);
    ASSERT_TRUE(s.ok()) << s.message;

    // Read
    auto cloud_in = read_point_cloud(path, PointCloudFormat::PCR_Binary);
    ASSERT_NE(cloud_in, nullptr);

    // Verify
    EXPECT_EQ(cloud_in->count(), 100);
    EXPECT_TRUE(cloud_in->has_channel("intensity"));

    for (size_t i = 0; i < 100; ++i) {
        EXPECT_DOUBLE_EQ(cloud_in->x()[i], x[i]);
        EXPECT_DOUBLE_EQ(cloud_in->y()[i], y[i]);
    }

    const float* intensity_in = cloud_in->channel_f32("intensity");
    for (size_t i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(intensity_in[i], intensity_out[i]);
    }
}

TEST_F(PointCloudIOTest, PCRBinary_MultipleChannels) {
    std::string path = test_dir_ + "multi_channel.pcrp";

    auto cloud_out = PointCloud::create(50, MemoryLocation::Host);
    cloud_out->resize(50);

    // Fill coordinates
    double* x = const_cast<double*>(cloud_out->x());
    double* y = const_cast<double*>(cloud_out->y());
    for (size_t i = 0; i < 50; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i) * 2.0;
    }

    // Add multiple channels of different types
    cloud_out->add_channel("elevation", DataType::Float32);
    cloud_out->add_channel("classification", DataType::Int32);

    float* elev = cloud_out->channel_f32("elevation");
    int32_t* classif = cloud_out->channel_i32("classification");

    for (size_t i = 0; i < 50; ++i) {
        elev[i] = static_cast<float>(i) * 100.0f;
        classif[i] = static_cast<int32_t>(i % 10);
    }

    // Write and read
    Status s = write_point_cloud(path, *cloud_out);
    ASSERT_TRUE(s.ok());

    auto cloud_in = read_point_cloud(path);
    ASSERT_NE(cloud_in, nullptr);
    EXPECT_EQ(cloud_in->count(), 50);
    EXPECT_TRUE(cloud_in->has_channel("elevation"));
    EXPECT_TRUE(cloud_in->has_channel("classification"));

    // Verify data
    const float* elev_in = cloud_in->channel_f32("elevation");
    const int32_t* classif_in = cloud_in->channel_i32("classification");

    for (size_t i = 0; i < 50; ++i) {
        EXPECT_FLOAT_EQ(elev_in[i], elev[i]);
        EXPECT_EQ(classif_in[i], classif[i]);
    }
}

TEST_F(PointCloudIOTest, PCRBinary_ReadInfo) {
    std::string path = test_dir_ + "info.pcrp";

    auto cloud_out = PointCloud::create(200, MemoryLocation::Host);
    cloud_out->resize(200);
    cloud_out->set_crs(CRS::from_epsg(4326));
    cloud_out->add_channel("value", DataType::Float32);

    Status s = write_point_cloud(path, *cloud_out);
    ASSERT_TRUE(s.ok());

    // Read info without loading data
    PointCloudInfo info;
    s = read_point_cloud_info(path, info);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_EQ(info.num_points, 200);
    EXPECT_EQ(info.channels.size(), 1);
    EXPECT_EQ(info.channels[0].name, "value");
    EXPECT_EQ(info.channels[0].dtype, DataType::Float32);
    EXPECT_TRUE(info.crs.is_valid());
}

TEST_F(PointCloudIOTest, DISABLED_PCRBinary_EmptyCloud) {
    std::string path = test_dir_ + "empty.pcrp";

    auto cloud_out = PointCloud::create(0, MemoryLocation::Host);

    Status s = write_point_cloud(path, *cloud_out);
    ASSERT_TRUE(s.ok());

    auto cloud_in = read_point_cloud(path);
    ASSERT_NE(cloud_in, nullptr);
    EXPECT_EQ(cloud_in->count(), 0);
}

// ===========================================================================
// CSV Format Tests
// ===========================================================================

TEST_F(PointCloudIOTest, CSV_WriteReadRoundTrip) {
    std::string path = test_dir_ + "test.csv";

    auto cloud_out = PointCloud::create(20, MemoryLocation::Host);
    cloud_out->resize(20);

    double* x = const_cast<double*>(cloud_out->x());
    double* y = const_cast<double*>(cloud_out->y());
    for (size_t i = 0; i < 20; ++i) {
        x[i] = static_cast<double>(i) * 1.5;
        y[i] = static_cast<double>(i) * 2.5;
    }

    cloud_out->add_channel("z", DataType::Float64);
    void* z_data = cloud_out->channel_data("z");
    double* z = static_cast<double*>(z_data);
    for (size_t i = 0; i < 20; ++i) {
        z[i] = static_cast<double>(i) * 3.5;
    }

    // Write CSV
    Status s = write_point_cloud(path, *cloud_out, PointCloudFormat::CSV);
    ASSERT_TRUE(s.ok()) << s.message;

    // Read CSV
    auto cloud_in = read_point_cloud(path, PointCloudFormat::CSV);
    ASSERT_NE(cloud_in, nullptr);
    EXPECT_EQ(cloud_in->count(), 20);
    EXPECT_TRUE(cloud_in->has_channel("z"));

    // Verify data (CSV has some precision loss)
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_NEAR(cloud_in->x()[i], x[i], 1e-10);
        EXPECT_NEAR(cloud_in->y()[i], y[i], 1e-10);
    }

    void* z_in_data = cloud_in->channel_data("z");
    double* z_in = static_cast<double*>(z_in_data);
    for (size_t i = 0; i < 20; ++i) {
        EXPECT_NEAR(z_in[i], z[i], 1e-10);
    }
}

TEST_F(PointCloudIOTest, CSV_ReadInfo) {
    std::string path = test_dir_ + "info.csv";

    auto cloud_out = PointCloud::create(30, MemoryLocation::Host);
    cloud_out->resize(30);
    cloud_out->add_channel("intensity", DataType::Float64);
    cloud_out->add_channel("classification", DataType::Float64);

    Status s = write_point_cloud(path, *cloud_out, PointCloudFormat::CSV);
    ASSERT_TRUE(s.ok());

    PointCloudInfo info;
    s = read_point_cloud_info(path, info, PointCloudFormat::CSV);
    ASSERT_TRUE(s.ok());

    EXPECT_EQ(info.num_points, 30);
    EXPECT_EQ(info.channels.size(), 2);
}

// ===========================================================================
// Auto Format Detection Tests
// ===========================================================================

TEST_F(PointCloudIOTest, AutoDetect_PCRBinary) {
    std::string path = test_dir_ + "auto.pcrp";

    auto cloud_out = PointCloud::create(10, MemoryLocation::Host);
    cloud_out->resize(10);
    Status s = write_point_cloud(path, *cloud_out, PointCloudFormat::PCR_Binary);
    ASSERT_TRUE(s.ok());

    // Read with auto-detect
    auto cloud_in = read_point_cloud(path, PointCloudFormat::Auto);
    ASSERT_NE(cloud_in, nullptr);
    EXPECT_EQ(cloud_in->count(), 10);
}

TEST_F(PointCloudIOTest, AutoDetect_CSV) {
    std::string path = test_dir_ + "auto.csv";

    auto cloud_out = PointCloud::create(10, MemoryLocation::Host);
    cloud_out->resize(10);

    // Fill with minimal data
    double* x = const_cast<double*>(cloud_out->x());
    double* y = const_cast<double*>(cloud_out->y());
    for (int i = 0; i < 10; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i);
    }

    Status s = write_point_cloud(path, *cloud_out, PointCloudFormat::CSV);
    ASSERT_TRUE(s.ok());

    // Read with auto-detect
    auto cloud_in = read_point_cloud(path, PointCloudFormat::Auto);
    ASSERT_NE(cloud_in, nullptr);
    EXPECT_EQ(cloud_in->count(), 10);
}

// ===========================================================================
// Streaming Reader Tests
// ===========================================================================

TEST_F(PointCloudIOTest, StreamingReader_PCRBinary) {
    std::string path = test_dir_ + "stream.pcrp";

    // Create a larger point cloud
    auto cloud_out = PointCloud::create(1000, MemoryLocation::Host);
    cloud_out->resize(1000);
    cloud_out->add_channel("value", DataType::Float32);

    double* x = const_cast<double*>(cloud_out->x());
    double* y = const_cast<double*>(cloud_out->y());
    float* val = cloud_out->channel_f32("value");

    for (size_t i = 0; i < 1000; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i) * 2.0;
        val[i] = static_cast<float>(i) * 0.1f;
    }

    Status s = write_point_cloud(path, *cloud_out);
    ASSERT_TRUE(s.ok());

    // Open reader
    auto reader = PointCloudReader::open(path);
    ASSERT_NE(reader, nullptr);

    const PointCloudInfo& info = reader->info();
    EXPECT_EQ(info.num_points, 1000);
    EXPECT_EQ(info.channels.size(), 1);

    // Read in chunks
    auto chunk = PointCloud::create(100, MemoryLocation::Host);
    size_t total_read = 0;

    while (!reader->eof()) {
        size_t n = reader->read_chunk(*chunk, 100);
        if (n == 0) break;
        total_read += n;
    }

    EXPECT_EQ(total_read, 1000);
}

TEST_F(PointCloudIOTest, StreamingReader_CSV) {
    std::string path = test_dir_ + "stream.csv";

    auto cloud_out = PointCloud::create(500, MemoryLocation::Host);
    cloud_out->resize(500);

    // Fill with data
    double* x = const_cast<double*>(cloud_out->x());
    double* y = const_cast<double*>(cloud_out->y());
    for (size_t i = 0; i < 500; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i) * 2.0;
    }

    cloud_out->add_channel("z", DataType::Float64);
    void* z_data = cloud_out->channel_data("z");
    double* z = static_cast<double*>(z_data);
    for (size_t i = 0; i < 500; ++i) {
        z[i] = static_cast<double>(i) * 3.0;
    }

    Status s = write_point_cloud(path, *cloud_out, PointCloudFormat::CSV);
    ASSERT_TRUE(s.ok());

    auto reader = PointCloudReader::open(path, PointCloudFormat::CSV);
    ASSERT_NE(reader, nullptr);
    EXPECT_EQ(reader->info().num_points, 500);

    auto chunk = PointCloud::create(50, MemoryLocation::Host);
    size_t total = 0;

    while (!reader->eof()) {
        size_t n = reader->read_chunk(*chunk, 50);
        if (n == 0) break;
        total += n;
    }

    EXPECT_EQ(total, 500);
}

TEST_F(PointCloudIOTest, StreamingReader_Rewind) {
    std::string path = test_dir_ + "rewind.pcrp";

    auto cloud_out = PointCloud::create(100, MemoryLocation::Host);
    cloud_out->resize(100);
    Status s = write_point_cloud(path, *cloud_out);
    ASSERT_TRUE(s.ok());

    auto reader = PointCloudReader::open(path);
    ASSERT_NE(reader, nullptr);

    auto chunk = PointCloud::create(100, MemoryLocation::Host);

    // Read all
    size_t n1 = reader->read_chunk(*chunk, 100);
    EXPECT_EQ(n1, 100);
    EXPECT_TRUE(reader->eof());

    // Rewind
    Status rewind_status = reader->rewind();
    ASSERT_TRUE(rewind_status.ok());
    EXPECT_FALSE(reader->eof());

    // Read again
    size_t n2 = reader->read_chunk(*chunk, 100);
    EXPECT_EQ(n2, 100);
}

// ===========================================================================
// Error Handling Tests
// ===========================================================================

TEST_F(PointCloudIOTest, ReadNonexistentFile) {
    std::string path = test_dir_ + "nonexistent.pcrp";

    auto cloud = read_point_cloud(path);
    EXPECT_EQ(cloud, nullptr);

    PointCloudInfo info;
    Status s = read_point_cloud_info(path, info);
    EXPECT_FALSE(s.ok());
}

TEST_F(PointCloudIOTest, WriteInvalidLocation) {
    std::string path = test_dir_ + "invalid.pcrp";

    auto cloud = PointCloud::create(10, MemoryLocation::Host);
    // Try to pretend it's on device (would fail if we could set it)
    // For now just test that Host works
    Status s = write_point_cloud(path, *cloud);
    EXPECT_TRUE(s.ok());
}

TEST_F(PointCloudIOTest, CorruptedPCRBinary) {
    std::string path = test_dir_ + "corrupt.pcrp";

    // Write garbage
    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    uint32_t bad_magic = 0xDEADBEEF;
    fwrite(&bad_magic, 4, 1, f);
    fclose(f);

    auto cloud = read_point_cloud(path, PointCloudFormat::PCR_Binary);
    EXPECT_EQ(cloud, nullptr);
}

TEST_F(PointCloudIOTest, LAS_NotImplemented) {
    std::string path = test_dir_ + "test.las";

    auto cloud_out = PointCloud::create(10, MemoryLocation::Host);
    Status s = write_point_cloud(path, *cloud_out, PointCloudFormat::LAS);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::NotImplemented);
}

// ===========================================================================
// Special Values Tests
// ===========================================================================

TEST_F(PointCloudIOTest, SpecialFloatValues_PCRBinary) {
    std::string path = test_dir_ + "special.pcrp";

    auto cloud_out = PointCloud::create(5, MemoryLocation::Host);
    cloud_out->resize(5);
    cloud_out->add_channel("value", DataType::Float32);

    double* x = const_cast<double*>(cloud_out->x());
    double* y = const_cast<double*>(cloud_out->y());
    float* val = cloud_out->channel_f32("value");

    x[0] = 0.0;
    x[1] = -0.0;
    x[2] = std::numeric_limits<double>::infinity();
    x[3] = -std::numeric_limits<double>::infinity();
    x[4] = std::numeric_limits<double>::quiet_NaN();

    for (int i = 0; i < 5; ++i) {
        y[i] = static_cast<double>(i);
    }

    val[0] = 1.0f;
    val[1] = std::numeric_limits<float>::infinity();
    val[2] = -std::numeric_limits<float>::infinity();
    val[3] = std::nanf("");
    val[4] = 42.0f;

    Status s = write_point_cloud(path, *cloud_out);
    ASSERT_TRUE(s.ok());

    auto cloud_in = read_point_cloud(path);
    ASSERT_NE(cloud_in, nullptr);

    const double* x_in = cloud_in->x();
    EXPECT_DOUBLE_EQ(x_in[0], 0.0);
    EXPECT_TRUE(std::isinf(x_in[2]) && x_in[2] > 0);
    EXPECT_TRUE(std::isinf(x_in[3]) && x_in[3] < 0);
    EXPECT_TRUE(std::isnan(x_in[4]));

    const float* val_in = cloud_in->channel_f32("value");
    EXPECT_TRUE(std::isinf(val_in[1]) && val_in[1] > 0);
    EXPECT_TRUE(std::isinf(val_in[2]) && val_in[2] < 0);
    EXPECT_TRUE(std::isnan(val_in[3]));
}

TEST_F(PointCloudIOTest, DISABLED_SpecialFloatValues_CSV) {
    std::string path = test_dir_ + "special.csv";

    auto cloud_out = PointCloud::create(3, MemoryLocation::Host);
    cloud_out->resize(3);
    cloud_out->add_channel("value", DataType::Float64);

    double* x = const_cast<double*>(cloud_out->x());
    x[0] = std::numeric_limits<double>::infinity();
    x[1] = -std::numeric_limits<double>::infinity();
    x[2] = std::numeric_limits<double>::quiet_NaN();

    Status s = write_point_cloud(path, *cloud_out, PointCloudFormat::CSV);
    ASSERT_TRUE(s.ok());

    auto cloud_in = read_point_cloud(path, PointCloudFormat::CSV);
    ASSERT_NE(cloud_in, nullptr);

    const double* x_in = cloud_in->x();
    EXPECT_TRUE(std::isinf(x_in[0]) && x_in[0] > 0);
    EXPECT_TRUE(std::isinf(x_in[1]) && x_in[1] < 0);
    EXPECT_TRUE(std::isnan(x_in[2]));
}

// ===========================================================================
// CRS Preservation Tests
// ===========================================================================

TEST_F(PointCloudIOTest, CRS_Preservation_PCRBinary) {
    std::string path = test_dir_ + "crs.pcrp";

    auto cloud_out = PointCloud::create(10, MemoryLocation::Host);
    cloud_out->resize(10);
    cloud_out->set_crs(CRS::from_epsg(32610));  // UTM Zone 10N

    Status s = write_point_cloud(path, *cloud_out);
    ASSERT_TRUE(s.ok());

    auto cloud_in = read_point_cloud(path);
    ASSERT_NE(cloud_in, nullptr);

    CRS crs_in = cloud_in->crs();
    EXPECT_TRUE(crs_in.is_valid());
    // WKT is preserved, EPSG code would need to be extracted from WKT
    EXPECT_FALSE(crs_in.wkt.empty());
}

TEST_F(PointCloudIOTest, LargePointCloud) {
    std::string path = test_dir_ + "large.pcrp";

    // Create 100K points
    auto cloud_out = PointCloud::create(100000, MemoryLocation::Host);
    cloud_out->resize(100000);
    cloud_out->add_channel("intensity", DataType::Float32);

    Status s = write_point_cloud(path, *cloud_out);
    ASSERT_TRUE(s.ok());

    PointCloudInfo info;
    s = read_point_cloud_info(path, info);
    ASSERT_TRUE(s.ok());
    EXPECT_EQ(info.num_points, 100000);
}
