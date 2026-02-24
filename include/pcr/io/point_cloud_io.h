#pragma once

#include "pcr/core/types.h"
#include "pcr/core/point_cloud.h"
#include <string>
#include <memory>
#include <vector>

namespace pcr {

// ---------------------------------------------------------------------------
// Point cloud file formats
// ---------------------------------------------------------------------------
enum class PointCloudFormat : uint8_t {
    PCR_Binary,     // native SoA binary format (fast, no dependencies)
    CSV,            // text CSV with header row
    LAS,            // LAS 1.2-1.4 (via LASlib or PDAL)
    LAZ,            // compressed LAS
    Auto            // detect from file extension
};

// ---------------------------------------------------------------------------
// Native PCR binary format
//
//   Header:
//     magic:          uint32 "PCRP"
//     version:        uint32 1
//     num_points:     uint64
//     num_channels:   uint32
//     crs_wkt_len:    uint32
//     crs_wkt:        char[crs_wkt_len]
//     channel_table:  { name_len: uint16, name: char[], dtype: uint8 } × num_channels
//
//   Body (SoA):
//     x:              float64[num_points]
//     y:              float64[num_points]
//     channels:       <dtype>[num_points] × num_channels, in channel_table order
//
// Designed for fast mmap-based loading.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// Read a point cloud from file. Format auto-detected if Auto.
/// Returns a host-memory PointCloud.
std::unique_ptr<PointCloud> read_point_cloud(const std::string& path,
                                             PointCloudFormat format = PointCloudFormat::Auto);

/// Read only the header/metadata (point count, channels, CRS) without loading data.
struct PointCloudInfo {
    size_t                     num_points;
    std::vector<ChannelDesc>   channels;
    CRS                        crs;
    BBox                       bounds;    // may be empty if format doesn't store it
};

Status read_point_cloud_info(const std::string& path,
                             PointCloudInfo& info,
                             PointCloudFormat format = PointCloudFormat::Auto);

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write a PointCloud to file. Cloud must be on Host.
Status write_point_cloud(const std::string& path,
                         const PointCloud& cloud,
                         PointCloudFormat format = PointCloudFormat::PCR_Binary);

// ---------------------------------------------------------------------------
// Streaming reader — for large files that don't fit in memory
// ---------------------------------------------------------------------------
class PointCloudReader {
public:
    ~PointCloudReader();

    static std::unique_ptr<PointCloudReader> open(const std::string& path,
                                                  PointCloudFormat format = PointCloudFormat::Auto);

    /// Get metadata without reading points.
    const PointCloudInfo& info() const;

    /// Read next chunk of up to `max_points` into `cloud`.
    /// Returns number of points read (0 when EOF).
    /// `cloud` must be pre-allocated with sufficient capacity.
    size_t read_chunk(PointCloud& cloud, size_t max_points);

    /// Reset to beginning of file.
    Status rewind();

    /// Whether all points have been read.
    bool eof() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcr
