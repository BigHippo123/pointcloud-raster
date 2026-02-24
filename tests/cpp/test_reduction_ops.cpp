#include "pcr/ops/builtin_ops.h"
#include "pcr/ops/reduction_registry.h"
#include "test_helpers.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace pcr;

// ===========================================================================
// Direct Op Tests — test each op's methods directly
// ===========================================================================

TEST(SumOpTest, IdentityCombineMerge) {
    auto id = SumOp::identity();
    EXPECT_FLOAT_EQ(id.sum, 0.0f);

    auto s1 = SumOp::combine(id, 10.0f);
    EXPECT_FLOAT_EQ(s1.sum, 10.0f);

    auto s2 = SumOp::combine(s1, 20.0f);
    EXPECT_FLOAT_EQ(s2.sum, 30.0f);

    auto s3 = SumOp::merge(s2, s1);
    EXPECT_FLOAT_EQ(s3.sum, 40.0f);

    float final_val = SumOp::finalize(s3);
    EXPECT_FLOAT_EQ(final_val, 40.0f);
}

TEST(MaxOpTest, IdentityCombineMerge) {
    auto id = MaxOp::identity();
    EXPECT_FLOAT_EQ(id.val, -FLT_MAX);

    auto s1 = MaxOp::combine(id, 10.0f);
    EXPECT_FLOAT_EQ(s1.val, 10.0f);

    auto s2 = MaxOp::combine(s1, 5.0f);
    EXPECT_FLOAT_EQ(s2.val, 10.0f);  // max stays at 10

    auto s3 = MaxOp::combine(s2, 15.0f);
    EXPECT_FLOAT_EQ(s3.val, 15.0f);

    auto s4 = MaxOp::merge(s3, s1);
    EXPECT_FLOAT_EQ(s4.val, 15.0f);

    float final_val = MaxOp::finalize(s4);
    EXPECT_FLOAT_EQ(final_val, 15.0f);
}

TEST(MaxOpTest, FinalizeEmptyReturnsNaN) {
    auto id = MaxOp::identity();
    float final_val = MaxOp::finalize(id);
    EXPECT_TRUE(std::isnan(final_val));
}

TEST(MinOpTest, IdentityCombineMerge) {
    auto id = MinOp::identity();
    EXPECT_FLOAT_EQ(id.val, FLT_MAX);

    auto s1 = MinOp::combine(id, 10.0f);
    EXPECT_FLOAT_EQ(s1.val, 10.0f);

    auto s2 = MinOp::combine(s1, 15.0f);
    EXPECT_FLOAT_EQ(s2.val, 10.0f);  // min stays at 10

    auto s3 = MinOp::combine(s2, 5.0f);
    EXPECT_FLOAT_EQ(s3.val, 5.0f);

    float final_val = MinOp::finalize(s3);
    EXPECT_FLOAT_EQ(final_val, 5.0f);
}

TEST(MinOpTest, FinalizeEmptyReturnsNaN) {
    auto id = MinOp::identity();
    float final_val = MinOp::finalize(id);
    EXPECT_TRUE(std::isnan(final_val));
}

TEST(CountOpTest, IdentityCombineMerge) {
    auto id = CountOp::identity();
    EXPECT_FLOAT_EQ(id.count, 0.0f);

    auto s1 = CountOp::combine(id, 999.0f);  // value ignored
    EXPECT_FLOAT_EQ(s1.count, 1.0f);

    auto s2 = CountOp::combine(s1, 123.0f);  // value ignored
    EXPECT_FLOAT_EQ(s2.count, 2.0f);

    auto s3 = CountOp::merge(s2, s1);
    EXPECT_FLOAT_EQ(s3.count, 3.0f);

    float final_val = CountOp::finalize(s3);
    EXPECT_FLOAT_EQ(final_val, 3.0f);
}

TEST(CountOpTest, FinalizeEmptyReturnsNaN) {
    auto id = CountOp::identity();
    float final_val = CountOp::finalize(id);
    EXPECT_TRUE(std::isnan(final_val));
}

TEST(AverageOpTest, IdentityCombineMerge) {
    auto id = AverageOp::identity();
    EXPECT_FLOAT_EQ(id.sum, 0.0f);
    EXPECT_FLOAT_EQ(id.count, 0.0f);

    auto s1 = AverageOp::combine(id, 10.0f);
    EXPECT_FLOAT_EQ(s1.sum, 10.0f);
    EXPECT_FLOAT_EQ(s1.count, 1.0f);

    auto s2 = AverageOp::combine(s1, 20.0f);
    EXPECT_FLOAT_EQ(s2.sum, 30.0f);
    EXPECT_FLOAT_EQ(s2.count, 2.0f);

    float final_val = AverageOp::finalize(s2);
    EXPECT_FLOAT_EQ(final_val, 15.0f);
}

TEST(AverageOpTest, Merge) {
    auto s1 = AverageOp::combine(AverageOp::identity(), 10.0f);
    s1 = AverageOp::combine(s1, 20.0f);  // avg=15, count=2

    auto s2 = AverageOp::combine(AverageOp::identity(), 30.0f);
    s2 = AverageOp::combine(s2, 40.0f);  // avg=35, count=2

    auto merged = AverageOp::merge(s1, s2);
    EXPECT_FLOAT_EQ(merged.sum, 100.0f);
    EXPECT_FLOAT_EQ(merged.count, 4.0f);
    EXPECT_FLOAT_EQ(AverageOp::finalize(merged), 25.0f);
}

TEST(AverageOpTest, FinalizeEmptyReturnsNaN) {
    auto id = AverageOp::identity();
    float final_val = AverageOp::finalize(id);
    EXPECT_TRUE(std::isnan(final_val));
}

// ===========================================================================
// Pack/Unpack Tests
// ===========================================================================

TEST(PackUnpackTest, SumOp) {
    std::vector<float> buffer(10, 0.0f);
    SumOp::State state{42.0f};

    pack_state<SumOp>(state, buffer.data(), 3, 10);
    EXPECT_FLOAT_EQ(buffer[3], 42.0f);

    SumOp::State unpacked = unpack_state<SumOp>(buffer.data(), 3, 10);
    EXPECT_FLOAT_EQ(unpacked.sum, 42.0f);
}

TEST(PackUnpackTest, AverageOp) {
    // AverageOp has 2 state floats: sum, count
    // Band-sequential layout: sum at [0..n-1], count at [n..2n-1]
    std::vector<float> buffer(20, 0.0f);  // 2 * 10 cells
    AverageOp::State state{100.0f, 5.0f};

    pack_state<AverageOp>(state, buffer.data(), 3, 10);
    EXPECT_FLOAT_EQ(buffer[3], 100.0f);      // sum field
    EXPECT_FLOAT_EQ(buffer[10 + 3], 5.0f);   // count field

    AverageOp::State unpacked = unpack_state<AverageOp>(buffer.data(), 3, 10);
    EXPECT_FLOAT_EQ(unpacked.sum, 100.0f);
    EXPECT_FLOAT_EQ(unpacked.count, 5.0f);
}

// ===========================================================================
// Registry Tests
// ===========================================================================

TEST(RegistryTest, GetReductionSum) {
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->type, ReductionType::Sum);
    EXPECT_EQ(info->state_floats, 1);
}

TEST(RegistryTest, GetReductionAverage) {
    const ReductionInfo* info = get_reduction(ReductionType::Average);
    ASSERT_NE(info, nullptr);
    EXPECT_EQ(info->type, ReductionType::Average);
    EXPECT_EQ(info->state_floats, 2);
}

TEST(RegistryTest, GetReductionUnimplemented) {
    // WeightedAverage is not yet registered (needs special kernel)
    const ReductionInfo* info = get_reduction(ReductionType::WeightedAverage);
    EXPECT_EQ(info, nullptr);

    // Median not implemented yet
    info = get_reduction(ReductionType::Median);
    EXPECT_EQ(info, nullptr);
}

TEST(RegistryTest, StateFloats) {
    EXPECT_EQ(reduction_state_floats(ReductionType::Sum), 1);
    EXPECT_EQ(reduction_state_floats(ReductionType::Max), 1);
    EXPECT_EQ(reduction_state_floats(ReductionType::Min), 1);
    EXPECT_EQ(reduction_state_floats(ReductionType::Count), 1);
    EXPECT_EQ(reduction_state_floats(ReductionType::Average), 2);

    // Unknown/unimplemented defaults to 1
    EXPECT_EQ(reduction_state_floats(ReductionType::Median), 1);
}

// ===========================================================================
// CPU Implementation Tests — test init/accumulate/merge/finalize via registry
// ===========================================================================

TEST(CPUAccumulateTest, SumOp) {
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    // 3x3 grid, 9 cells
    const int64_t tile_cells = 9;
    std::vector<float> state(tile_cells, 0.0f);

    // Initialize state
    Status s = info->init_state(state.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok());
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(state[i], 0.0f);
    }

    // Accumulate some points
    // cell_indices: which cell each point falls into
    // values: measurement value for each point
    std::vector<uint32_t> cell_indices = {0, 0, 1, 2, 2, 2, 4};
    std::vector<float>    values        = {10, 20, 30, 5, 5, 5, 100};

    s = info->accumulate(
        cell_indices.data(),
        values.data(),
        state.data(),
        7,
        tile_cells,
        nullptr);
    ASSERT_TRUE(s.ok());

    // Check accumulated state
    EXPECT_FLOAT_EQ(state[0], 30.0f);   // 10 + 20
    EXPECT_FLOAT_EQ(state[1], 30.0f);   // 30
    EXPECT_FLOAT_EQ(state[2], 15.0f);   // 5 + 5 + 5
    EXPECT_FLOAT_EQ(state[3], 0.0f);    // no points
    EXPECT_FLOAT_EQ(state[4], 100.0f);  // 100
}

TEST(CPUAccumulateTest, AverageOp) {
    const ReductionInfo* info = get_reduction(ReductionType::Average);
    ASSERT_NE(info, nullptr);

    const int64_t tile_cells = 5;
    std::vector<float> state(tile_cells * 2, 0.0f);  // 2 floats per cell

    Status s = info->init_state(state.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok());

    // Accumulate points into cell 0 and cell 1
    std::vector<uint32_t> cell_indices = {0, 0, 0, 1, 1};
    std::vector<float>    values        = {10, 20, 30, 50, 50};

    s = info->accumulate(
        cell_indices.data(),
        values.data(),
        state.data(),
        5,
        tile_cells,
        nullptr);
    ASSERT_TRUE(s.ok());

    // Finalize to get average values
    std::vector<float> output(tile_cells, 0.0f);
    s = info->finalize(state.data(), output.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok());

    EXPECT_FLOAT_EQ(output[0], 20.0f);  // (10+20+30)/3
    EXPECT_FLOAT_EQ(output[1], 50.0f);  // (50+50)/2
    EXPECT_TRUE(std::isnan(output[2])); // no points
    EXPECT_TRUE(std::isnan(output[3]));
    EXPECT_TRUE(std::isnan(output[4]));
}

TEST(CPUMergeTest, SumOp) {
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    const int64_t tile_cells = 4;

    // State 1: [10, 20, 30, 40]
    std::vector<float> state1(tile_cells);
    for (int i = 0; i < 4; ++i) {
        state1[i] = (i + 1) * 10.0f;
    }

    // State 2: [5, 5, 5, 5]
    std::vector<float> state2(tile_cells, 5.0f);

    // Merge state2 into state1
    Status s = info->merge_state(state1.data(), state2.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok());

    EXPECT_FLOAT_EQ(state1[0], 15.0f);
    EXPECT_FLOAT_EQ(state1[1], 25.0f);
    EXPECT_FLOAT_EQ(state1[2], 35.0f);
    EXPECT_FLOAT_EQ(state1[3], 45.0f);
}

TEST(CPUMergeTest, MaxOp) {
    const ReductionInfo* info = get_reduction(ReductionType::Max);
    ASSERT_NE(info, nullptr);

    const int64_t tile_cells = 3;

    std::vector<float> state1 = {10.0f, 50.0f, 30.0f};
    std::vector<float> state2 = {20.0f, 40.0f, 60.0f};

    Status s = info->merge_state(state1.data(), state2.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok());

    EXPECT_FLOAT_EQ(state1[0], 20.0f);  // max(10, 20)
    EXPECT_FLOAT_EQ(state1[1], 50.0f);  // max(50, 40)
    EXPECT_FLOAT_EQ(state1[2], 60.0f);  // max(30, 60)
}

TEST(CPUFinalizeTest, CountOp) {
    const ReductionInfo* info = get_reduction(ReductionType::Count);
    ASSERT_NE(info, nullptr);

    const int64_t tile_cells = 4;
    std::vector<float> state = {0.0f, 1.0f, 5.0f, 100.0f};
    std::vector<float> output(tile_cells);

    Status s = info->finalize(state.data(), output.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok());

    EXPECT_TRUE(std::isnan(output[0]));  // 0 count → NaN
    EXPECT_FLOAT_EQ(output[1], 1.0f);
    EXPECT_FLOAT_EQ(output[2], 5.0f);
    EXPECT_FLOAT_EQ(output[3], 100.0f);
}

// ===========================================================================
// Edge Cases
// ===========================================================================

TEST(EdgeCaseTest, EmptyAccumulation) {
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    const int64_t tile_cells = 5;
    std::vector<float> state(tile_cells);

    info->init_state(state.data(), tile_cells, nullptr);

    // Accumulate zero points
    std::vector<uint32_t> empty_indices;
    std::vector<float>    empty_values;

    Status s = info->accumulate(
        empty_indices.data(),
        empty_values.data(),
        state.data(),
        0,
        tile_cells,
        nullptr);

    EXPECT_TRUE(s.ok());

    // State should remain at identity
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(state[i], 0.0f);
    }
}

TEST(EdgeCaseTest, SingleCell) {
    const ReductionInfo* info = get_reduction(ReductionType::Average);
    ASSERT_NE(info, nullptr);

    const int64_t tile_cells = 1;
    std::vector<float> state(2, 0.0f);  // 2 floats for Average

    info->init_state(state.data(), tile_cells, nullptr);

    std::vector<uint32_t> cell_indices = {0, 0, 0};
    std::vector<float>    values        = {100, 200, 300};

    info->accumulate(cell_indices.data(), values.data(), state.data(), 3, tile_cells, nullptr);

    std::vector<float> output(1);
    info->finalize(state.data(), output.data(), tile_cells, nullptr);

    EXPECT_FLOAT_EQ(output[0], 200.0f);  // (100+200+300)/3
}

TEST(EdgeCaseTest, CellIndexOutOfRange) {
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    const int64_t tile_cells = 5;
    std::vector<float> state(tile_cells, 0.0f);

    info->init_state(state.data(), tile_cells, nullptr);

    std::vector<uint32_t> cell_indices = {0, 1, 10};  // 10 is out of range
    std::vector<float>    values        = {1, 2, 3};

    Status s = info->accumulate(
        cell_indices.data(),
        values.data(),
        state.data(),
        3,
        tile_cells,
        nullptr);

    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code, StatusCode::InvalidArgument);
}
