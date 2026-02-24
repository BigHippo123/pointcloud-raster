#include "pcr/engine/accumulator.h"
#include "pcr/core/grid.h"
#include "pcr/ops/reduction_registry.h"
#include "pcr/ops/builtin_ops.h"
#include "pcr/engine/memory_pool.h"

#ifdef PCR_HAS_CUDA
#include "pcr/engine/accumulator_kernels.h"
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// Accumulator implementation
// ---------------------------------------------------------------------------
struct Accumulator::Impl {
    MemoryPool* pool;

    explicit Impl(MemoryPool* p) : pool(p) {}
};

Accumulator::~Accumulator() = default;

std::unique_ptr<Accumulator> Accumulator::create(MemoryPool* pool) {
    auto acc = std::unique_ptr<Accumulator>(new Accumulator);
    acc->impl_ = std::make_unique<Impl>(pool);
    return acc;
}

// ---------------------------------------------------------------------------
// CPU implementation (always available)
// ---------------------------------------------------------------------------
static Status accumulate_cpu(ReductionType type,
                             const TileBatch& batch,
                             float* state,
                             int64_t tile_cells,
                             void* stream) {
    // Check memory location
    if (batch.location != MemoryLocation::Host) {
        return Status::error(StatusCode::InvalidArgument,
            "CPU accumulate requires Host memory location");
    }

    // Get reduction info from registry
    const ReductionInfo* info = get_reduction(type);
    if (!info) {
        return Status::error(StatusCode::InvalidArgument,
            "accumulate: reduction type not registered");
    }

    // Validate batch
    if (!batch.local_cell_indices || !batch.values || batch.num_points == 0) {
        return Status::success();  // Nothing to accumulate
    }

    // Use the registry's accumulate function
    return info->accumulate(batch.local_cell_indices, batch.values, state,
                           batch.num_points, tile_cells, stream);
}

template <typename Op>
static Status accumulate_cpu_typed(const TileBatch& batch,
                                   float* state,
                                   int64_t tile_cells,
                                   void* /*stream*/) {
    // Check memory location
    if (batch.location != MemoryLocation::Host) {
        return Status::error(StatusCode::InvalidArgument,
            "CPU accumulate requires Host memory location");
    }

    // Validate batch
    if (!batch.local_cell_indices || !batch.values || batch.num_points == 0) {
        return Status::success();  // Nothing to accumulate
    }

    // Simple CPU loop: for each point, unpack state, combine, pack back
    for (size_t i = 0; i < batch.num_points; ++i) {
        uint32_t cell = batch.local_cell_indices[i];
        float value = batch.values[i];

        // Bounds check
        if (cell >= static_cast<uint32_t>(tile_cells)) {
            return Status::error(StatusCode::InvalidArgument,
                "accumulate: cell index out of bounds");
        }

        // Unpack current state
        typename Op::State acc = unpack_state<Op>(state, cell, tile_cells);

        // Combine with new value
        acc = Op::combine(acc, value);

        // Pack back to state buffer
        pack_state<Op>(acc, state, cell, tile_cells);
    }

    return Status::success();
}

#ifdef PCR_HAS_CUDA

// Map template Op types to ReductionType enum
template <typename Op> struct OpToReductionType;
template <> struct OpToReductionType<SumOp>     { static constexpr ReductionType value = ReductionType::Sum; };
template <> struct OpToReductionType<MaxOp>     { static constexpr ReductionType value = ReductionType::Max; };
template <> struct OpToReductionType<MinOp>     { static constexpr ReductionType value = ReductionType::Min; };
template <> struct OpToReductionType<CountOp>   { static constexpr ReductionType value = ReductionType::Count; };
template <> struct OpToReductionType<AverageOp> { static constexpr ReductionType value = ReductionType::Average; };

// ---------------------------------------------------------------------------
// GPU implementation
// ---------------------------------------------------------------------------
static Status accumulate_gpu_dispatch(ReductionType type,
                                      const TileBatch& batch,
                                      float* state,
                                      int64_t tile_cells,
                                      MemoryPool* pool,
                                      void* stream) {
    // Check memory location
    if (batch.location != MemoryLocation::Device &&
        batch.location != MemoryLocation::HostPinned) {
        return Status::error(StatusCode::InvalidArgument,
            "GPU accumulate requires Device or HostPinned memory location");
    }

    // Validate batch
    if (!batch.local_cell_indices || !batch.values || batch.num_points == 0) {
        return Status::success();  // Nothing to accumulate
    }

    // Call GPU kernel
    return accumulate_gpu(
        type,
        batch.local_cell_indices,
        batch.values,
        batch.weights,
        batch.timestamps,
        batch.num_points,
        state,
        tile_cells,
        pool,
        stream);
}

// Public API: dispatcher based on memory location
Status Accumulator::accumulate(ReductionType type,
                                const TileBatch& batch,
                                float* state,
                                int64_t tile_cells,
                                void* stream) {
    // Dispatch based on memory location
    if (batch.location == MemoryLocation::Host) {
        return accumulate_cpu(type, batch, state, tile_cells, stream);
    } else {
        return accumulate_gpu_dispatch(type, batch, state, tile_cells, impl_->pool, stream);
    }
}

template <typename Op>
Status Accumulator::accumulate(const TileBatch& batch,
                                float* state,
                                int64_t tile_cells,
                                void* stream) {
    // Dispatch based on memory location
    if (batch.location == MemoryLocation::Host) {
        return accumulate_cpu_typed<Op>(batch, state, tile_cells, stream);
    } else {
        ReductionType type = OpToReductionType<Op>::value;
        return accumulate_gpu_dispatch(type, batch, state, tile_cells, impl_->pool, stream);
    }
}

// Explicit template instantiations for builtin ops (CUDA path)
template Status Accumulator::accumulate<SumOp>(const TileBatch&, float*, int64_t, void*);
template Status Accumulator::accumulate<MaxOp>(const TileBatch&, float*, int64_t, void*);
template Status Accumulator::accumulate<MinOp>(const TileBatch&, float*, int64_t, void*);
template Status Accumulator::accumulate<CountOp>(const TileBatch&, float*, int64_t, void*);
template Status Accumulator::accumulate<AverageOp>(const TileBatch&, float*, int64_t, void*);

#else

// ---------------------------------------------------------------------------
// CPU-only build: just use the CPU implementation
// ---------------------------------------------------------------------------
Status Accumulator::accumulate(ReductionType type,
                                const TileBatch& batch,
                                float* state,
                                int64_t tile_cells,
                                void* stream) {
    return accumulate_cpu(type, batch, state, tile_cells, stream);
}

template <typename Op>
Status Accumulator::accumulate(const TileBatch& batch,
                                float* state,
                                int64_t tile_cells,
                                void* stream) {
    return accumulate_cpu_typed<Op>(batch, state, tile_cells, stream);
}

// Explicit template instantiations for builtin ops (CPU-only build)
template Status Accumulator::accumulate<SumOp>(const TileBatch&, float*, int64_t, void*);
template Status Accumulator::accumulate<MaxOp>(const TileBatch&, float*, int64_t, void*);
template Status Accumulator::accumulate<MinOp>(const TileBatch&, float*, int64_t, void*);
template Status Accumulator::accumulate<CountOp>(const TileBatch&, float*, int64_t, void*);
template Status Accumulator::accumulate<AverageOp>(const TileBatch&, float*, int64_t, void*);

#endif // PCR_HAS_CUDA

} // namespace pcr
