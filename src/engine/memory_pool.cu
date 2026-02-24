#include "pcr/engine/memory_pool.h"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

namespace pcr {

// ---------------------------------------------------------------------------
// Simple memory pool implementation using CUDA memory
// ---------------------------------------------------------------------------
struct MemoryPool::Impl {
    void* base_ptr = nullptr;
    size_t total_size = 0;
    size_t offset = 0;
    size_t high_water = 0;
};

MemoryPool::~MemoryPool() {
    if (impl_ && impl_->base_ptr) {
        cudaFree(impl_->base_ptr);
    }
}

std::unique_ptr<MemoryPool> MemoryPool::create(size_t size_bytes) {
    auto pool = std::unique_ptr<MemoryPool>(new MemoryPool);
    pool->impl_ = std::make_unique<Impl>();
    pool->impl_->total_size = size_bytes;

    cudaError_t err = cudaMalloc(&pool->impl_->base_ptr, size_bytes);
    if (err != cudaSuccess || !pool->impl_->base_ptr) {
        return nullptr;
    }

    return pool;
}

void* MemoryPool::allocate(size_t size) {
    if (!impl_ || !impl_->base_ptr) return nullptr;

    // Align to 256 bytes
    constexpr size_t ALIGN = 256;
    size_t aligned_size = ((size + ALIGN - 1) / ALIGN) * ALIGN;

    if (impl_->offset + aligned_size > impl_->total_size) {
        return nullptr;  // Pool exhausted
    }

    void* ptr = static_cast<char*>(impl_->base_ptr) + impl_->offset;
    impl_->offset += aligned_size;
    impl_->high_water = std::max(impl_->high_water, impl_->offset);

    return ptr;
}

void MemoryPool::reset() {
    if (impl_) {
        impl_->offset = 0;
    }
}

size_t MemoryPool::bytes_used() const {
    return impl_ ? impl_->high_water : 0;
}

size_t MemoryPool::bytes_total() const {
    return impl_ ? impl_->total_size : 0;
}

size_t MemoryPool::bytes_available() const {
    return impl_ ? (impl_->total_size - impl_->offset) : 0;
}

} // namespace pcr
