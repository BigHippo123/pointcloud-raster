#include "pcr/core/types.h"
#include "test_helpers.h"
#include <gtest/gtest.h>
#include <cmath>

using namespace pcr;

// ===========================================================================
// BBox Tests
// ===========================================================================

TEST(BBoxTest, DefaultConstructionIsInvalid) {
    BBox bbox;
    EXPECT_FALSE(bbox.valid());
}

TEST(BBoxTest, ExpandWithSinglePoint) {
    BBox bbox;
    bbox.expand(10.0, 20.0);

    EXPECT_TRUE(bbox.valid());
    EXPECT_DOUBLE_EQ(bbox.min_x, 10.0);
    EXPECT_DOUBLE_EQ(bbox.min_y, 20.0);
    EXPECT_DOUBLE_EQ(bbox.max_x, 10.0);
    EXPECT_DOUBLE_EQ(bbox.max_y, 20.0);
}

TEST(BBoxTest, ExpandWithMultiplePoints) {
    BBox bbox;
    bbox.expand(10.0, 20.0);
    bbox.expand(30.0, 40.0);
    bbox.expand(5.0, 15.0);

    EXPECT_TRUE(bbox.valid());
    EXPECT_DOUBLE_EQ(bbox.min_x, 5.0);
    EXPECT_DOUBLE_EQ(bbox.min_y, 15.0);
    EXPECT_DOUBLE_EQ(bbox.max_x, 30.0);
    EXPECT_DOUBLE_EQ(bbox.max_y, 40.0);
}

TEST(BBoxTest, ExpandWithBBox) {
    BBox bbox1;
    bbox1.expand(0.0, 0.0);
    bbox1.expand(10.0, 10.0);

    BBox bbox2;
    bbox2.expand(5.0, 5.0);
    bbox2.expand(20.0, 20.0);

    bbox1.expand(bbox2);

    EXPECT_DOUBLE_EQ(bbox1.min_x, 0.0);
    EXPECT_DOUBLE_EQ(bbox1.min_y, 0.0);
    EXPECT_DOUBLE_EQ(bbox1.max_x, 20.0);
    EXPECT_DOUBLE_EQ(bbox1.max_y, 20.0);
}

TEST(BBoxTest, Contains) {
    BBox bbox;
    bbox.expand(0.0, 0.0);
    bbox.expand(10.0, 10.0);

    EXPECT_TRUE(bbox.contains(5.0, 5.0));      // inside
    EXPECT_TRUE(bbox.contains(0.0, 0.0));      // min corner
    EXPECT_TRUE(bbox.contains(10.0, 10.0));    // max corner
    EXPECT_FALSE(bbox.contains(-1.0, 5.0));    // outside x
    EXPECT_FALSE(bbox.contains(5.0, -1.0));    // outside y
    EXPECT_FALSE(bbox.contains(11.0, 5.0));    // outside x
    EXPECT_FALSE(bbox.contains(5.0, 11.0));    // outside y
}

TEST(BBoxTest, WidthHeight) {
    BBox bbox;
    bbox.expand(10.0, 20.0);
    bbox.expand(50.0, 80.0);

    EXPECT_DOUBLE_EQ(bbox.width(), 40.0);
    EXPECT_DOUBLE_EQ(bbox.height(), 60.0);
}

// ===========================================================================
// CRS Tests
// ===========================================================================

TEST(CRSTest, CreateFromEPSG) {
    CRS crs = CRS::from_epsg(4326);

    EXPECT_TRUE(crs.is_valid());
    EXPECT_EQ(crs.epsg, 4326);
    EXPECT_FALSE(crs.wkt.empty());
}

TEST(CRSTest, CreateFromWKT) {
    // Simple WKT for WGS84
    std::string wkt = "GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563]],PRIMEM[\"Greenwich\",0],UNIT[\"degree\",0.0174532925199433]]";

    CRS crs = CRS::from_wkt(wkt);

    EXPECT_TRUE(crs.is_valid());
    EXPECT_FALSE(crs.wkt.empty());
}

TEST(CRSTest, IsGeographic) {
    CRS crs = CRS::from_epsg(4326);  // WGS84 - geographic
    EXPECT_TRUE(crs.is_geographic());
    EXPECT_FALSE(crs.is_projected());
}

TEST(CRSTest, IsProjected) {
    CRS crs = CRS::from_epsg(3857);  // Web Mercator - projected
    EXPECT_TRUE(crs.is_projected());
    // Note: Some PROJ versions may detect 3857 as both projected and geographic
    // due to its derivation from WGS84, so we only test is_projected()
}

TEST(CRSTest, EquivalentToSame) {
    CRS crs1 = CRS::from_epsg(4326);
    CRS crs2 = CRS::from_epsg(4326);

    EXPECT_TRUE(crs1.equivalent_to(crs2));
}

TEST(CRSTest, EquivalentToDifferent) {
    CRS crs1 = CRS::from_epsg(4326);  // WGS84
    CRS crs2 = CRS::from_epsg(3857);  // Web Mercator

    EXPECT_FALSE(crs1.equivalent_to(crs2));
}

// ===========================================================================
// DataType Tests
// ===========================================================================

TEST(DataTypeTest, TypeSizes) {
    EXPECT_EQ(data_type_size(DataType::Float32), 4);
    EXPECT_EQ(data_type_size(DataType::Float64), 8);
    EXPECT_EQ(data_type_size(DataType::Int32), 4);
    EXPECT_EQ(data_type_size(DataType::UInt32), 4);
    EXPECT_EQ(data_type_size(DataType::Int16), 2);
    EXPECT_EQ(data_type_size(DataType::UInt16), 2);
    EXPECT_EQ(data_type_size(DataType::UInt8), 1);
}

// ===========================================================================
// TileIndex Tests
// ===========================================================================

TEST(TileIndexTest, Equality) {
    TileIndex t1{0, 0};
    TileIndex t2{0, 0};
    TileIndex t3{1, 0};

    EXPECT_TRUE(t1 == t2);
    EXPECT_FALSE(t1 == t3);
}

TEST(TileIndexTest, Ordering) {
    TileIndex t1{0, 0};
    TileIndex t2{0, 1};
    TileIndex t3{1, 0};

    EXPECT_TRUE(t1 < t2);
    EXPECT_TRUE(t1 < t3);
    EXPECT_FALSE(t2 < t1);
}

// ===========================================================================
// Status Tests
// ===========================================================================

TEST(StatusTest, Success) {
    Status s = Status::success();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code, StatusCode::Ok);
}

TEST(StatusTest, Error) {
    Status s = Status::error(StatusCode::InvalidArgument, "test error");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
    EXPECT_EQ(s.message, "test error");
}

// ===========================================================================
// NoDataPolicy Tests
// ===========================================================================

TEST(NoDataPolicyTest, DefaultIsNaN) {
    NoDataPolicy policy;
    EXPECT_TRUE(policy.use_nan);
    EXPECT_TRUE(std::isnan(policy.sentinel()));
}

TEST(NoDataPolicyTest, CustomValue) {
    NoDataPolicy policy;
    policy.use_nan = false;
    policy.value = -9999.0f;

    EXPECT_FALSE(policy.use_nan);
    EXPECT_FLOAT_EQ(policy.sentinel(), -9999.0f);
}
