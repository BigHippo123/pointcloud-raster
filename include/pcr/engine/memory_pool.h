#pragma once

#include "pcr/core/types.h"
#include <cstddef>
#include <memory>

namespace pcr {

// ---------------------------------------------------------------------------
// MemoryPool — Arena allocator for GPU temporary buffers.
//
// Avoids cudaMalloc/cudaFree overhead per accumulate() call.
// Allocations are sub-allocated from a pre-allocated pool.
// reset() reclaims all allocations without freeing the underlying memory.
//
// NOT thread-safe — use one pool per CUDA stream.
// ---------------------------------------------------------------------------
class MemoryPool {
public:
    ~MemoryPool();

    /// Create a pool with `size_bytes` of device memory.
    static std::unique_ptr<MemoryPool> create(size_t size_bytes);

    /// Sub-allocate `size` bytes (aligned to 256 bytes). Returns device pointer.
    /// Returns nullptr if pool is exhausted.
    void* allocate(size_t size);

    /// Typed convenience
    template <typename T>
    T* allocate_array(size_t count) {
        return static_cast<T*>(allocate(count * sizeof(T)));
    }

    /// Reclaim all sub-allocations. Does NOT free underlying memory.
    void reset();

    /// Current bytes allocated (high-water mark since last reset).
    size_t bytes_used() const;

    /// Total pool capacity.
    size_t bytes_total() const;

    /// Bytes remaining.
    size_t bytes_available() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcr
