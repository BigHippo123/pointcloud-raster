#include <gtest/gtest.h>
#include "pcr/engine/memory_pool.h"
#include <cuda_runtime.h>

using namespace pcr;

class MemoryPoolGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
    }
};

TEST_F(MemoryPoolGPUTest, CreateAndAccessors) {
    auto pool = MemoryPool::create(1024 * 1024);  // 1MB
    ASSERT_NE(pool, nullptr);

    EXPECT_EQ(pool->bytes_total(), 1024u * 1024);
    EXPECT_EQ(pool->bytes_used(), 0u);
    EXPECT_EQ(pool->bytes_available(), 1024u * 1024);
}

TEST_F(MemoryPoolGPUTest, BasicAllocation) {
    auto pool = MemoryPool::create(1024 * 1024);
    ASSERT_NE(pool, nullptr);

    void* p1 = pool->allocate(256);
    ASSERT_NE(p1, nullptr);
    EXPECT_GE(pool->bytes_used(), 256u);

    void* p2 = pool->allocate(512);
    ASSERT_NE(p2, nullptr);
    EXPECT_NE(p1, p2);

    // Pointers should not overlap
    EXPECT_GE(static_cast<char*>(p2) - static_cast<char*>(p1), 256);
}

TEST_F(MemoryPoolGPUTest, Alignment256Bytes) {
    auto pool = MemoryPool::create(1024 * 1024);
    ASSERT_NE(pool, nullptr);

    void* p1 = pool->allocate(1);    // 1 byte, but next alloc should be 256-aligned
    void* p2 = pool->allocate(100);

    uintptr_t addr = reinterpret_cast<uintptr_t>(p2);
    EXPECT_EQ(addr % 256, 0u) << "Allocation should be 256-byte aligned";
}

TEST_F(MemoryPoolGPUTest, ExhaustionReturnsNull) {
    auto pool = MemoryPool::create(1024);  // 1KB
    ASSERT_NE(pool, nullptr);

    void* p = pool->allocate(2048);  // More than pool size
    EXPECT_EQ(p, nullptr);
}

TEST_F(MemoryPoolGPUTest, Reset) {
    auto pool = MemoryPool::create(4096);
    ASSERT_NE(pool, nullptr);

    void* p1 = pool->allocate(1024);
    ASSERT_NE(p1, nullptr);
    EXPECT_GE(pool->bytes_used(), 1024u);

    pool->reset();
    EXPECT_EQ(pool->bytes_used(), 0u);
    EXPECT_EQ(pool->bytes_available(), 4096u);

    // Should be able to allocate again after reset
    void* p2 = pool->allocate(2048);
    ASSERT_NE(p2, nullptr);
}

TEST_F(MemoryPoolGPUTest, MultipleCycles) {
    auto pool = MemoryPool::create(1024 * 1024);
    ASSERT_NE(pool, nullptr);

    for (int cycle = 0; cycle < 10; ++cycle) {
        void* p1 = pool->allocate(10000);
        ASSERT_NE(p1, nullptr) << "cycle " << cycle;

        void* p2 = pool->allocate(20000);
        ASSERT_NE(p2, nullptr) << "cycle " << cycle;

        pool->reset();
    }
}

TEST_F(MemoryPoolGPUTest, TypedAllocation) {
    auto pool = MemoryPool::create(1024 * 1024);
    ASSERT_NE(pool, nullptr);

    float* fp = pool->allocate_array<float>(1000);
    ASSERT_NE(fp, nullptr);

    uint32_t* ip = pool->allocate_array<uint32_t>(500);
    ASSERT_NE(ip, nullptr);

    // Write to device memory via kernel (just verify no crash)
    cudaMemset(fp, 0, 1000 * sizeof(float));
    cudaMemset(ip, 0, 500 * sizeof(uint32_t));
    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST_F(MemoryPoolGPUTest, ZeroSizeAlloc) {
    auto pool = MemoryPool::create(1024);
    ASSERT_NE(pool, nullptr);

    void* p = pool->allocate(0);
    EXPECT_EQ(p, nullptr);
}
