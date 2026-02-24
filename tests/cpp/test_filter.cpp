#include <gtest/gtest.h>
#include "pcr/engine/filter.h"
#include "pcr/core/point_cloud.h"

using namespace pcr;

class FilterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a test point cloud with 100 points
        cloud = PointCloud::create(100, MemoryLocation::Host);
        cloud->resize(100);

        // Add test channels
        cloud->add_channel("intensity", DataType::Float32);
        cloud->add_channel("classification", DataType::Float32);

        // Fill with test data
        auto* intensity = cloud->channel_f32("intensity");
        auto* classification = cloud->channel_f32("classification");

        for (size_t i = 0; i < 100; ++i) {
            intensity[i] = static_cast<float>(i);
            classification[i] = static_cast<float>(i % 5);  // Values: 0, 1, 2, 3, 4
        }
    }

    std::unique_ptr<PointCloud> cloud;
};

TEST_F(FilterTest, EmptySpec_PassesAllPoints) {
    FilterSpec spec;

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(count, 100);

    // Check that indices are 0..99
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(indices[i], i);
    }

    free(indices);
}

TEST_F(FilterTest, Equal) {
    FilterSpec spec;
    spec.add("classification", CompareOp::Equal, 2.0f);

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(count, 20);  // 20 points with classification == 2

    // Verify all passing points have classification == 2
    auto* classification = cloud->channel_f32("classification");
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(classification[indices[i]], 2.0f);
    }

    free(indices);
}

TEST_F(FilterTest, Less) {
    FilterSpec spec;
    spec.add("intensity", CompareOp::Less, 10.0f);

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(count, 10);  // Points 0..9

    auto* intensity = cloud->channel_f32("intensity");
    for (size_t i = 0; i < count; ++i) {
        EXPECT_LT(intensity[indices[i]], 10.0f);
    }

    free(indices);
}

TEST_F(FilterTest, GreaterEqual) {
    FilterSpec spec;
    spec.add("intensity", CompareOp::GreaterEqual, 90.0f);

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(count, 10);  // Points 90..99

    auto* intensity = cloud->channel_f32("intensity");
    for (size_t i = 0; i < count; ++i) {
        EXPECT_GE(intensity[indices[i]], 90.0f);
    }

    free(indices);
}

TEST_F(FilterTest, InSet) {
    FilterSpec spec;
    spec.add_in_set("classification", {1.0f, 3.0f});

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(count, 40);  // 20 points with class 1 + 20 with class 3

    auto* classification = cloud->channel_f32("classification");
    for (size_t i = 0; i < count; ++i) {
        float cls = classification[indices[i]];
        EXPECT_TRUE(cls == 1.0f || cls == 3.0f);
    }

    free(indices);
}

TEST_F(FilterTest, NotInSet) {
    FilterSpec spec;
    FilterPredicate pred;
    pred.channel_name = "classification";
    pred.op = CompareOp::NotInSet;
    pred.value_set = {0.0f, 1.0f};
    spec.predicates.push_back(pred);

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(count, 60);  // All except class 0 and 1

    auto* classification = cloud->channel_f32("classification");
    for (size_t i = 0; i < count; ++i) {
        float cls = classification[indices[i]];
        EXPECT_TRUE(cls != 0.0f && cls != 1.0f);
    }

    free(indices);
}

TEST_F(FilterTest, MultiplePredicates_AND) {
    FilterSpec spec;
    spec.add("intensity", CompareOp::GreaterEqual, 50.0f);
    spec.add("intensity", CompareOp::Less, 60.0f);
    spec.add("classification", CompareOp::Equal, 0.0f);

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    ASSERT_TRUE(s.ok()) << s.message;

    // Points 50..59 with class 0 (only 50 and 55 satisfy this)
    EXPECT_EQ(count, 2);

    auto* intensity = cloud->channel_f32("intensity");
    auto* classification = cloud->channel_f32("classification");
    for (size_t i = 0; i < count; ++i) {
        float inten = intensity[indices[i]];
        float cls = classification[indices[i]];
        EXPECT_GE(inten, 50.0f);
        EXPECT_LT(inten, 60.0f);
        EXPECT_EQ(cls, 0.0f);
    }

    free(indices);
}

TEST_F(FilterTest, NoMatches) {
    FilterSpec spec;
    spec.add("intensity", CompareOp::Greater, 1000.0f);

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    ASSERT_TRUE(s.ok()) << s.message;
    EXPECT_EQ(count, 0);

    free(indices);
}

TEST_F(FilterTest, InvalidChannel) {
    FilterSpec spec;
    spec.add("nonexistent", CompareOp::Equal, 0.0f);

    uint32_t* indices = nullptr;
    size_t count = 0;

    Status s = filter_points(*cloud, spec, &indices, &count);

    EXPECT_FALSE(s.ok());
    EXPECT_TRUE(s.message.find("channel not found") != std::string::npos);
}
