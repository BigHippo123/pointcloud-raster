#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <limits>
#include <cmath>

#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// Scalar type support — channels can hold different types
// ---------------------------------------------------------------------------
enum class DataType : uint8_t {
    Float32,
    Float64,
    Int32,
    UInt32,
    Int16,
    UInt16,
    UInt8
};

/// Size in bytes for a given DataType
size_t data_type_size(DataType dt);

// ---------------------------------------------------------------------------
// Reduction operations
// ---------------------------------------------------------------------------
enum class ReductionType : uint8_t {
    Sum,
    Max,
    Min,
    Average,
    WeightedAverage,
    Count,
    Median,
    Percentile,       // requires percentile parameter
    MostRecent,       // requires timestamp channel
    PriorityMerge,    // keeps value with highest priority channel value
    Custom            // user-provided functor (template path only)
};

// ---------------------------------------------------------------------------
// Axis-aligned bounding box (2D, double precision for geo coords)
// ---------------------------------------------------------------------------
struct BBox {
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double max_y = std::numeric_limits<double>::lowest();

    void expand(double x, double y);
    void expand(const BBox& other);
    bool contains(double x, double y) const;

    double width()  const { return max_x - min_x; }
    double height() const { return max_y - min_y; }
    bool   valid()  const { return max_x >= min_x && max_y >= min_y; }
};

// ---------------------------------------------------------------------------
// Coordinate Reference System — thin wrapper around PROJ/WKT
// ---------------------------------------------------------------------------
struct CRS {
    std::string wkt;          // WKT2 string (authoritative)
    int         epsg = 0;     // EPSG code if known, 0 = unknown

    bool is_projected() const;
    bool is_geographic() const;
    bool is_valid() const { return !wkt.empty() || epsg != 0; }

    static CRS from_epsg(int code);
    static CRS from_wkt(const std::string& wkt_str);

    bool equivalent_to(const CRS& other) const;
};

// ---------------------------------------------------------------------------
// Nodata policy
// ---------------------------------------------------------------------------
struct NoDataPolicy {
    float value = std::nanf("");
    bool  use_nan = true;

    float sentinel() const { return use_nan ? std::nanf("") : value; }
};

// ---------------------------------------------------------------------------
// Memory location tag
// ---------------------------------------------------------------------------
enum class MemoryLocation : uint8_t {
    Host,
    HostPinned,
    Device
};

// ---------------------------------------------------------------------------
// Tile index (row, col within tile grid)
// ---------------------------------------------------------------------------
struct TileIndex {
    int row = 0;
    int col = 0;

    bool operator==(const TileIndex& o) const { return row == o.row && col == o.col; }
    bool operator<(const TileIndex& o) const {
        return row < o.row || (row == o.row && col < o.col);
    }
};

// ---------------------------------------------------------------------------
// Status / error reporting
// ---------------------------------------------------------------------------
enum class StatusCode : uint8_t {
    Ok,
    InvalidArgument,
    OutOfMemory,
    CudaError,
    IoError,
    CrsError,
    NotImplemented
};

struct Status {
    StatusCode  code = StatusCode::Ok;
    std::string message;

    bool ok() const { return code == StatusCode::Ok; }
    static Status success() { return {}; }
    static Status error(StatusCode c, const std::string& msg) { return {c, msg}; }
};

// ---------------------------------------------------------------------------
// CUDA error checking helper
// ---------------------------------------------------------------------------
#ifdef PCR_HAS_CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            return Status::error(StatusCode::CudaError, \
                std::string("CUDA error: ") + cudaGetErrorString(err)); \
        } \
    } while (0)
#endif

// ---------------------------------------------------------------------------
// GPU Capability Detection and Error Handling
// ---------------------------------------------------------------------------

/// Check if CUDA is available (compiled with CUDA support)
inline bool cuda_is_compiled() {
#ifdef PCR_HAS_CUDA
    return true;
#else
    return false;
#endif
}

/// Check if a CUDA-capable device is available at runtime
inline bool cuda_device_available() {
#ifdef PCR_HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

/// Get number of available CUDA devices
inline int cuda_device_count() {
#ifdef PCR_HAS_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return 0;
    }
    return device_count;
#else
    return 0;
#endif
}

/// Get GPU device name (for logging/debugging)
inline std::string cuda_device_name(int device_id = 0) {
#ifdef PCR_HAS_CUDA
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return "Unknown GPU";
    }
    return std::string(prop.name);
#else
    (void)device_id;  // Unused
    return "CUDA not compiled";
#endif
}

/// Get free and total GPU memory in bytes
inline bool cuda_get_memory_info(size_t* free_bytes, size_t* total_bytes, int device_id = 0) {
#ifdef PCR_HAS_CUDA
    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        return false;
    }
    err = cudaMemGetInfo(free_bytes, total_bytes);
    return (err == cudaSuccess);
#else
    (void)device_id;
    (void)free_bytes;
    (void)total_bytes;
    return false;
#endif
}

} // namespace pcr
