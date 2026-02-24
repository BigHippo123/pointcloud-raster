#pragma once

#include "pcr/core/types.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace pcr {

class PointCloud;
class MemoryPool;

// ---------------------------------------------------------------------------
// FilterPredicate — describes a filter condition on a metadata channel.
//
// Predicates are combined with AND logic. Each predicate operates on one
// channel and supports common comparison ops.
// ---------------------------------------------------------------------------
enum class CompareOp : uint8_t {
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    InSet,          // value is in a set of allowed values
    NotInSet
};

struct FilterPredicate {
    std::string channel_name;     // metadata channel to filter on
    CompareOp   op;
    float       value = 0.0f;    // for scalar comparisons
    std::vector<float> value_set; // for InSet / NotInSet
};

// ---------------------------------------------------------------------------
// FilterSpec — collection of predicates (AND-combined)
// ---------------------------------------------------------------------------
struct FilterSpec {
    std::vector<FilterPredicate> predicates;

    /// Convenience: add a simple comparison predicate
    FilterSpec& add(const std::string& channel, CompareOp op, float value);

    /// Convenience: add a set membership predicate
    FilterSpec& add_in_set(const std::string& channel, const std::vector<float>& values);

    bool empty() const { return predicates.empty(); }
};

// ---------------------------------------------------------------------------
// Filter execution (GPU)
//
// Applies FilterSpec to a PointCloud on device. Returns compacted indices
// of surviving points, and the count.
//
// Two modes:
//   1. In-place compaction: reorder the PointCloud arrays to remove filtered points.
//   2. Index-based: return an index array of surviving points (non-destructive).
// ---------------------------------------------------------------------------

/// Evaluate predicates on device PointCloud. Returns device array of surviving
/// point indices and their count. Caller must free the returned array (or use pool).
///
/// `out_indices` and `out_count` are set on return.
/// Allocates from `pool` if provided, else cudaMalloc.
Status filter_points(const PointCloud& cloud,
                     const FilterSpec& spec,
                     uint32_t** out_indices,     // device pointer, caller frees
                     size_t*    out_count,
                     MemoryPool* pool = nullptr,
                     void*      stream = nullptr);

} // namespace pcr
