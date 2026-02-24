#pragma once

#include "pcr/core/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace pcr {

// ---------------------------------------------------------------------------
// Channel descriptor — describes one named array in the SoA
// ---------------------------------------------------------------------------
struct ChannelDesc {
    std::string name;
    DataType    dtype    = DataType::Float32;
    size_t      offset   = 0;    // byte offset into channel storage (internal use)
};

// ---------------------------------------------------------------------------
// PointCloud — Structure-of-Arrays point cloud with named channels
//
// Coordinates (x, y) are always Float64 for geo precision.
// Value and metadata channels are registered by name and can be any DataType.
//
// Memory can reside on host or device. The cloud does NOT own external
// pointers passed via wrap() — caller must keep them alive.
// ---------------------------------------------------------------------------
class PointCloud {
public:
    // -- Construction -------------------------------------------------------

    PointCloud() = default;
    ~PointCloud();

    /// Create an empty cloud with capacity for `n` points.
    /// Allocates x, y on `loc`. Channels added later inherit same location.
    static std::unique_ptr<PointCloud> create(size_t capacity,
                                              MemoryLocation loc = MemoryLocation::Host);

    /// Wrap existing external SoA buffers (non-owning). Caller manages lifetime.
    /// `x` and `y` must each have `count` elements (Float64).
    static std::unique_ptr<PointCloud> wrap(double* x, double* y, size_t count,
                                            MemoryLocation loc = MemoryLocation::Host);

    // -- Channel management -------------------------------------------------

    /// Register a new value or metadata channel. Allocates storage.
    Status add_channel(const std::string& name, DataType dtype = DataType::Float32);

    /// Check if a channel exists
    bool has_channel(const std::string& name) const;

    /// Get channel descriptor (returns nullptr if not found)
    const ChannelDesc* channel(const std::string& name) const;

    /// Get list of all channel names
    std::vector<std::string> channel_names() const;

    // -- Raw data access (type-unsafe, low-level) ---------------------------

    double*       x();                 // world x coordinates
    const double* x() const;
    double*       y();                 // world y coordinates
    const double* y() const;

    /// Raw pointer to channel data. Caller must cast based on dtype.
    void*       channel_data(const std::string& name);
    const void* channel_data(const std::string& name) const;

    /// Typed convenience accessors (asserts dtype matches)
    float*       channel_f32(const std::string& name);
    const float* channel_f32(const std::string& name) const;
    int32_t*       channel_i32(const std::string& name);
    const int32_t* channel_i32(const std::string& name) const;

    // -- Properties ---------------------------------------------------------

    size_t         count()    const;   // number of points currently stored
    size_t         capacity() const;   // allocated capacity
    MemoryLocation location() const;
    CRS            crs()      const;
    void           set_crs(const CRS& crs);

    // -- Resize / append ----------------------------------------------------

    /// Resize point count (must be <= capacity). Does NOT reallocate.
    Status resize(size_t new_count);

    // -- Device transfer ----------------------------------------------------

    /// Copy entire cloud (coords + all channels) to a new MemoryLocation.
    /// Returns a new PointCloud; original is unchanged.
    std::unique_ptr<PointCloud> to(MemoryLocation dst) const;

    /// Copy to device asynchronously on given CUDA stream.
    /// Returned cloud is on Device; transfer may not be complete until stream syncs.
    std::unique_ptr<PointCloud> to_device_async(void* cuda_stream) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcr
