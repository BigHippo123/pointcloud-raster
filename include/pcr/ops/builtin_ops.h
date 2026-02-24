#pragma once

#include "pcr/ops/reduction_op.h"

namespace pcr {

// ---------------------------------------------------------------------------
// SumOp
// ---------------------------------------------------------------------------
struct SumOp {
    struct State { float sum; };

    static PCR_HD State identity()                    { return {0.0f}; }
    static PCR_HD State combine(State acc, float val) { return {acc.sum + val}; }
    static PCR_HD State merge(State a, State b)       { return {a.sum + b.sum}; }
    static PCR_HD float finalize(State acc)           { return acc.sum; }
    static constexpr int state_floats()               { return 1; }
};

// ---------------------------------------------------------------------------
// MaxOp
// ---------------------------------------------------------------------------
struct MaxOp {
    struct State { float val; };

    static PCR_HD State identity()                    { return {-FLT_MAX}; }
    static PCR_HD State combine(State acc, float val) { return {fmaxf(acc.val, val)}; }
    static PCR_HD State merge(State a, State b)       { return {fmaxf(a.val, b.val)}; }
    static PCR_HD float finalize(State acc)           { return acc.val == -FLT_MAX ? NAN : acc.val; }
    static constexpr int state_floats()               { return 1; }
};

// ---------------------------------------------------------------------------
// MinOp
// ---------------------------------------------------------------------------
struct MinOp {
    struct State { float val; };

    static PCR_HD State identity()                    { return {FLT_MAX}; }
    static PCR_HD State combine(State acc, float val) { return {fminf(acc.val, val)}; }
    static PCR_HD State merge(State a, State b)       { return {fminf(a.val, b.val)}; }
    static PCR_HD float finalize(State acc)           { return acc.val == FLT_MAX ? NAN : acc.val; }
    static constexpr int state_floats()               { return 1; }
};

// ---------------------------------------------------------------------------
// CountOp — counts points per cell, ignores value
// ---------------------------------------------------------------------------
struct CountOp {
    struct State { float count; };   // float for GPU compatibility

    static PCR_HD State identity()                    { return {0.0f}; }
    static PCR_HD State combine(State acc, float)     { return {acc.count + 1.0f}; }
    static PCR_HD State merge(State a, State b)       { return {a.count + b.count}; }
    static PCR_HD float finalize(State acc)           { return acc.count > 0.0f ? acc.count : NAN; }
    static constexpr int state_floats()               { return 1; }
};

// ---------------------------------------------------------------------------
// AverageOp — requires 2 state floats (sum + count)
// ---------------------------------------------------------------------------
struct AverageOp {
    struct State { float sum; float count; };

    static PCR_HD State identity()                    { return {0.0f, 0.0f}; }
    static PCR_HD State combine(State acc, float val) { return {acc.sum + val, acc.count + 1.0f}; }
    static PCR_HD State merge(State a, State b)       { return {a.sum + b.sum, a.count + b.count}; }
    static PCR_HD float finalize(State acc) {
        return acc.count > 0.0f ? acc.sum / acc.count : NAN;
    }
    static constexpr int state_floats()               { return 2; }
};

// ---------------------------------------------------------------------------
// WeightedAverageOp — point value is the measurement, weight from a second channel.
//   combine() takes the value; weight is injected separately via WeightedCombine.
//   For the sort-reduce path, we use a 2-element value: {val, weight}.
// ---------------------------------------------------------------------------
struct WeightedAverageOp {
    struct State { float weighted_sum; float weight_sum; };

    static PCR_HD State identity()                    { return {0.0f, 0.0f}; }

    /// Standard combine: val encodes value, weight passed separately.
    /// In the kernel, the caller packs (value * weight) into val and weight into weight_arg.
    static PCR_HD State combine_weighted(State acc, float val, float weight) {
        return {acc.weighted_sum + val * weight, acc.weight_sum + weight};
    }

    static PCR_HD State merge(State a, State b) {
        return {a.weighted_sum + b.weighted_sum, a.weight_sum + b.weight_sum};
    }

    static PCR_HD float finalize(State acc) {
        return acc.weight_sum > 0.0f ? acc.weighted_sum / acc.weight_sum : NAN;
    }
    static constexpr int state_floats()               { return 2; }
};

// ---------------------------------------------------------------------------
// MostRecentOp — keeps value with highest timestamp
//   State holds {value, timestamp}. Timestamp channel provided separately.
// ---------------------------------------------------------------------------
struct MostRecentOp {
    struct State { float value; float timestamp; };

    static PCR_HD State identity()                    { return {NAN, -FLT_MAX}; }

    static PCR_HD State combine_timestamped(State acc, float val, float ts) {
        return ts > acc.timestamp ? State{val, ts} : acc;
    }

    static PCR_HD State merge(State a, State b) {
        return a.timestamp >= b.timestamp ? a : b;
    }

    static PCR_HD float finalize(State acc)           { return acc.value; }
    static constexpr int state_floats()               { return 2; }
};

// ---------------------------------------------------------------------------
// pack_state / unpack_state specializations for builtins
// Band-sequential layout: field f stored at base[f * num_cells + i]
// ---------------------------------------------------------------------------

// 1-float state ops (Sum, Max, Min, Count)
#define PCR_PACK_UNPACK_1(OP)                                                   \
template <> inline PCR_HD void pack_state<OP>(                                  \
    const OP::State& s, float* base, int64_t i, int64_t) {                     \
    base[i] = s.val;                                                            \
}                                                                               \
template <> inline PCR_HD OP::State unpack_state<OP>(                           \
    const float* base, int64_t i, int64_t) {                                   \
    return {base[i]};                                                           \
}

// Sum and Count use different field names, handle manually
template <> inline PCR_HD void pack_state<SumOp>(
    const SumOp::State& s, float* base, int64_t i, int64_t) { base[i] = s.sum; }
template <> inline PCR_HD SumOp::State unpack_state<SumOp>(
    const float* base, int64_t i, int64_t) { return {base[i]}; }

template <> inline PCR_HD void pack_state<MaxOp>(
    const MaxOp::State& s, float* base, int64_t i, int64_t) { base[i] = s.val; }
template <> inline PCR_HD MaxOp::State unpack_state<MaxOp>(
    const float* base, int64_t i, int64_t) { return {base[i]}; }

template <> inline PCR_HD void pack_state<MinOp>(
    const MinOp::State& s, float* base, int64_t i, int64_t) { base[i] = s.val; }
template <> inline PCR_HD MinOp::State unpack_state<MinOp>(
    const float* base, int64_t i, int64_t) { return {base[i]}; }

template <> inline PCR_HD void pack_state<CountOp>(
    const CountOp::State& s, float* base, int64_t i, int64_t) { base[i] = s.count; }
template <> inline PCR_HD CountOp::State unpack_state<CountOp>(
    const float* base, int64_t i, int64_t) { return {base[i]}; }

// 2-float state ops (Average, WeightedAverage, MostRecent)
template <> inline PCR_HD void pack_state<AverageOp>(
    const AverageOp::State& s, float* base, int64_t i, int64_t n) {
    base[i] = s.sum; base[n + i] = s.count;
}
template <> inline PCR_HD AverageOp::State unpack_state<AverageOp>(
    const float* base, int64_t i, int64_t n) { return {base[i], base[n + i]}; }

template <> inline PCR_HD void pack_state<WeightedAverageOp>(
    const WeightedAverageOp::State& s, float* base, int64_t i, int64_t n) {
    base[i] = s.weighted_sum; base[n + i] = s.weight_sum;
}
template <> inline PCR_HD WeightedAverageOp::State unpack_state<WeightedAverageOp>(
    const float* base, int64_t i, int64_t n) { return {base[i], base[n + i]}; }

template <> inline PCR_HD void pack_state<MostRecentOp>(
    const MostRecentOp::State& s, float* base, int64_t i, int64_t n) {
    base[i] = s.value; base[n + i] = s.timestamp;
}
template <> inline PCR_HD MostRecentOp::State unpack_state<MostRecentOp>(
    const float* base, int64_t i, int64_t n) { return {base[i], base[n + i]}; }

} // namespace pcr
