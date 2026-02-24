#include <gtest/gtest.h>
#include "pcr/engine/accumulator.h"
#include "pcr/ops/reduction_registry.h"
#include "pcr/ops/builtin_ops.h"
#include <cstring>
#include <cmath>

using namespace pcr;

class AccumulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        acc = Accumulator::create();
    }

    std::unique_ptr<Accumulator> acc;
};

TEST_F(AccumulatorTest, Sum_SingleBatch) {
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    // Tile with 10 cells
    int64_t tile_cells = 10;
    std::vector<float> state(tile_cells, 0.0f);

    // Initialize state to identity
    Status s = info->init_state(state.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    // Batch: 5 points mapping to cells [0, 1, 2, 1, 0]
    uint32_t cell_indices[] = {0, 1, 2, 1, 0};
    float values[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = cell_indices;
    batch.values = values;
    batch.num_points = 5;

    s = acc->accumulate(ReductionType::Sum, batch, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // Expected: cell 0 = 60, cell 1 = 60, cell 2 = 30, rest = 0
    EXPECT_FLOAT_EQ(state[0], 60.0f);
    EXPECT_FLOAT_EQ(state[1], 60.0f);
    EXPECT_FLOAT_EQ(state[2], 30.0f);
    for (int i = 3; i < tile_cells; ++i) {
        EXPECT_FLOAT_EQ(state[i], 0.0f);
    }
}

TEST_F(AccumulatorTest, Max_SingleBatch) {
    const ReductionInfo* info = get_reduction(ReductionType::Max);
    ASSERT_NE(info, nullptr);

    int64_t tile_cells = 10;
    std::vector<float> state(tile_cells, -FLT_MAX);

    Status s = info->init_state(state.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    uint32_t cell_indices[] = {0, 1, 0, 1, 2};
    float values[] = {10.0f, 20.0f, 50.0f, 15.0f, 100.0f};

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = cell_indices;
    batch.values = values;
    batch.num_points = 5;

    s = acc->accumulate(ReductionType::Max, batch, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // Expected: cell 0 = 50, cell 1 = 20, cell 2 = 100
    EXPECT_FLOAT_EQ(state[0], 50.0f);
    EXPECT_FLOAT_EQ(state[1], 20.0f);
    EXPECT_FLOAT_EQ(state[2], 100.0f);
}

TEST_F(AccumulatorTest, Min_SingleBatch) {
    const ReductionInfo* info = get_reduction(ReductionType::Min);
    ASSERT_NE(info, nullptr);

    int64_t tile_cells = 10;
    std::vector<float> state(tile_cells, FLT_MAX);

    Status s = info->init_state(state.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    uint32_t cell_indices[] = {0, 1, 0, 1, 2};
    float values[] = {10.0f, 20.0f, 5.0f, 15.0f, 100.0f};

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = cell_indices;
    batch.values = values;
    batch.num_points = 5;

    s = acc->accumulate(ReductionType::Min, batch, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // Expected: cell 0 = 5, cell 1 = 15, cell 2 = 100
    EXPECT_FLOAT_EQ(state[0], 5.0f);
    EXPECT_FLOAT_EQ(state[1], 15.0f);
    EXPECT_FLOAT_EQ(state[2], 100.0f);
}

TEST_F(AccumulatorTest, Count_SingleBatch) {
    const ReductionInfo* info = get_reduction(ReductionType::Count);
    ASSERT_NE(info, nullptr);

    int64_t tile_cells = 10;
    std::vector<float> state(tile_cells, 0.0f);

    Status s = info->init_state(state.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    uint32_t cell_indices[] = {0, 0, 1, 1, 1, 2};
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = cell_indices;
    batch.values = values;
    batch.num_points = 6;

    s = acc->accumulate(ReductionType::Count, batch, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // Expected: cell 0 = 2, cell 1 = 3, cell 2 = 1
    EXPECT_FLOAT_EQ(state[0], 2.0f);
    EXPECT_FLOAT_EQ(state[1], 3.0f);
    EXPECT_FLOAT_EQ(state[2], 1.0f);
}

TEST_F(AccumulatorTest, Average_SingleBatch) {
    const ReductionInfo* info = get_reduction(ReductionType::Average);
    ASSERT_NE(info, nullptr);

    int64_t tile_cells = 10;
    int state_floats = info->state_floats;
    EXPECT_EQ(state_floats, 2);  // sum + count

    std::vector<float> state(state_floats * tile_cells, 0.0f);

    Status s = info->init_state(state.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    // Points: cell 0 gets {10, 20}, cell 1 gets {30}
    uint32_t cell_indices[] = {0, 1, 0};
    float values[] = {10.0f, 30.0f, 20.0f};

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = cell_indices;
    batch.values = values;
    batch.num_points = 3;

    s = acc->accumulate(ReductionType::Average, batch, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // State layout: [sum_0, sum_1, ..., sum_9, count_0, count_1, ..., count_9]
    // Cell 0: sum = 30, count = 2
    EXPECT_FLOAT_EQ(state[0], 30.0f);
    EXPECT_FLOAT_EQ(state[tile_cells + 0], 2.0f);

    // Cell 1: sum = 30, count = 1
    EXPECT_FLOAT_EQ(state[1], 30.0f);
    EXPECT_FLOAT_EQ(state[tile_cells + 1], 1.0f);

    // Finalize to get averages
    std::vector<float> output(tile_cells);
    s = info->finalize(state.data(), output.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_FLOAT_EQ(output[0], 15.0f);  // 30 / 2
    EXPECT_FLOAT_EQ(output[1], 30.0f);  // 30 / 1
}

TEST_F(AccumulatorTest, MultipleBatches_Accumulate) {
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    int64_t tile_cells = 5;
    std::vector<float> state(tile_cells, 0.0f);

    Status s = info->init_state(state.data(), tile_cells, nullptr);
    ASSERT_TRUE(s.ok()) << s.message;

    // First batch
    uint32_t cells1[] = {0, 1, 2};
    float vals1[] = {10.0f, 20.0f, 30.0f};

    TileBatch batch1;
    batch1.tile = {0, 0};
    batch1.local_cell_indices = cells1;
    batch1.values = vals1;
    batch1.num_points = 3;

    s = acc->accumulate(ReductionType::Sum, batch1, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // Second batch (same cells)
    uint32_t cells2[] = {0, 1, 2};
    float vals2[] = {5.0f, 10.0f, 15.0f};

    TileBatch batch2;
    batch2.tile = {0, 0};
    batch2.local_cell_indices = cells2;
    batch2.values = vals2;
    batch2.num_points = 3;

    s = acc->accumulate(ReductionType::Sum, batch2, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // Expected: accumulation across batches
    EXPECT_FLOAT_EQ(state[0], 15.0f);
    EXPECT_FLOAT_EQ(state[1], 30.0f);
    EXPECT_FLOAT_EQ(state[2], 45.0f);
}

TEST_F(AccumulatorTest, TemplateVersion_Sum) {
    int64_t tile_cells = 10;
    std::vector<float> state(tile_cells);

    // Initialize using pack_state
    for (int64_t i = 0; i < tile_cells; ++i) {
        pack_state<SumOp>(SumOp::identity(), state.data(), i, tile_cells);
    }

    uint32_t cell_indices[] = {0, 1, 0};
    float values[] = {10.0f, 20.0f, 30.0f};

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = cell_indices;
    batch.values = values;
    batch.num_points = 3;

    Status s = acc->accumulate<SumOp>(batch, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    EXPECT_FLOAT_EQ(state[0], 40.0f);
    EXPECT_FLOAT_EQ(state[1], 20.0f);
}

TEST_F(AccumulatorTest, TemplateVersion_Average) {
    int64_t tile_cells = 10;
    std::vector<float> state(AverageOp::state_floats() * tile_cells);

    // Initialize
    for (int64_t i = 0; i < tile_cells; ++i) {
        pack_state<AverageOp>(AverageOp::identity(), state.data(), i, tile_cells);
    }

    uint32_t cell_indices[] = {0, 0, 1};
    float values[] = {10.0f, 30.0f, 50.0f};

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = cell_indices;
    batch.values = values;
    batch.num_points = 3;

    Status s = acc->accumulate<AverageOp>(batch, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // Unpack and check
    auto state0 = unpack_state<AverageOp>(state.data(), 0, tile_cells);
    EXPECT_FLOAT_EQ(state0.sum, 40.0f);
    EXPECT_FLOAT_EQ(state0.count, 2.0f);

    auto state1 = unpack_state<AverageOp>(state.data(), 1, tile_cells);
    EXPECT_FLOAT_EQ(state1.sum, 50.0f);
    EXPECT_FLOAT_EQ(state1.count, 1.0f);

    // Finalize
    EXPECT_FLOAT_EQ(AverageOp::finalize(state0), 20.0f);
    EXPECT_FLOAT_EQ(AverageOp::finalize(state1), 50.0f);
}

TEST_F(AccumulatorTest, EmptyBatch) {
    const ReductionInfo* info = get_reduction(ReductionType::Sum);
    ASSERT_NE(info, nullptr);

    int64_t tile_cells = 10;
    std::vector<float> state(tile_cells, 0.0f);

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = nullptr;
    batch.values = nullptr;
    batch.num_points = 0;

    Status s = acc->accumulate(ReductionType::Sum, batch, state.data(), tile_cells);
    ASSERT_TRUE(s.ok()) << s.message;

    // State should be unchanged
    for (int i = 0; i < tile_cells; ++i) {
        EXPECT_FLOAT_EQ(state[i], 0.0f);
    }
}

TEST_F(AccumulatorTest, UnregisteredType) {
    int64_t tile_cells = 10;
    std::vector<float> state(tile_cells, 0.0f);

    uint32_t cell_indices[] = {0};
    float values[] = {10.0f};

    TileBatch batch;
    batch.tile = {0, 0};
    batch.local_cell_indices = cell_indices;
    batch.values = values;
    batch.num_points = 1;

    Status s = acc->accumulate(ReductionType::Custom, batch, state.data(), tile_cells);
    EXPECT_FALSE(s.ok());
    EXPECT_TRUE(s.message.find("not registered") != std::string::npos);
}
