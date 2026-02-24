#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

// This header is included in .cu files — all ops must be __host__ __device__.

#if defined(__CUDACC__) && defined(PCR_HAS_CUDA)
#define PCR_HD __host__ __device__
#else
#define PCR_HD
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// ReductionOp concept
//
// Every reduction op must provide:
//
//   struct State { ... };                           // per-cell accumulator
//   static PCR_HD State  identity();                // initial value before any points
//   static PCR_HD State  combine(State acc, float value);  // fold one point in
//   static PCR_HD State  merge(State a, State b);   // merge two accumulators (for grid merge)
//   static PCR_HD float  finalize(State acc);       // extract final output value
//   static constexpr int state_floats();            // number of floats in State (for serialization)
//
// The State must be trivially copyable and representable as N floats for
// GPU storage and disk serialization.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// State serialization helpers
//
// State is stored in the tile grid as a flat array of floats.
// For a grid with W*H cells and an op with state_floats() == K,
// the state buffer is K * W * H floats, stored band-interleaved-by-pixel:
//   [cell0_f0, cell0_f1, ..., cell0_fK-1, cell1_f0, ...]
//
// These helpers pack/unpack State to/from a float* at a given cell offset.
// ---------------------------------------------------------------------------

/// Pack State into float buffer at cell index `i`, stride = num_cells.
/// Layout: state field `f` is at base[f * num_cells + i]  (band-sequential)
template <typename Op>
PCR_HD void pack_state(const typename Op::State& s, float* base, int64_t i, int64_t num_cells);

/// Unpack State from float buffer at cell index `i`.
template <typename Op>
PCR_HD typename Op::State unpack_state(const float* base, int64_t i, int64_t num_cells);

// Note: pack_state and unpack_state must be specialized per Op.
// Builtin specializations are in builtin_ops.h.

} // namespace pcr
