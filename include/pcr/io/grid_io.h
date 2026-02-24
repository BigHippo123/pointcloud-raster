#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include <string>
#include <vector>
#include <memory>

namespace pcr {

class Grid;

// ---------------------------------------------------------------------------
// GeoTIFF write options
// ---------------------------------------------------------------------------
struct GeoTiffOptions {
    bool        cloud_optimized = false;    // COG: add overviews + tiling
    std::string compress        = "LZW";    // NONE, LZW, DEFLATE, ZSTD
    int         compress_level  = 6;        // for DEFLATE/ZSTD
    int         tile_width      = 256;      // internal TIFF tile width (0 = stripped)
    int         tile_height     = 256;      // internal TIFF tile height
    bool        bigtiff         = true;     // always use BigTIFF for >4GB files
    std::string overview_resampling = "average"; // NEAREST, AVERAGE, BILINEAR, etc.
};

// ---------------------------------------------------------------------------
// GeoTIFF writer — writes Grid to GeoTIFF file.
//
// Supports two modes:
//   1. Full grid: write a complete Grid object at once.
//   2. Tiled write: open file, write tiles incrementally, close.
// ---------------------------------------------------------------------------

/// Write a complete Grid to GeoTIFF. Grid must be on Host.
/// Band names are embedded as TIFF descriptions.
Status write_geotiff(const std::string& path,
                     const Grid& grid,
                     const GridConfig& config,
                     const GeoTiffOptions& options = {});

// ---------------------------------------------------------------------------
// Tiled GeoTIFF writer — for out-of-core assembly
// ---------------------------------------------------------------------------
class TiledGeoTiffWriter {
public:
    ~TiledGeoTiffWriter();

    /// Open file for writing. Creates the TIFF structure.
    /// `band_names` defines the output bands (one per reduction).
    static std::unique_ptr<TiledGeoTiffWriter> open(
        const std::string& path,
        const GridConfig& config,
        const std::vector<std::string>& band_names,
        const GeoTiffOptions& options = {});

    /// Write one tile's finalized data.
    /// `data` is host memory, band-sequential: data[band][row][col].
    /// Size per band: tile_cols * tile_rows floats.
    Status write_tile(TileIndex tile, const float* data, int num_bands);

    /// Finalize: write overviews (if COG), close file.
    Status close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// GeoTIFF reader — for loading existing rasters (e.g., reading back results)
// ---------------------------------------------------------------------------
Status read_geotiff_info(const std::string& path,
                         int& width, int& height, int& num_bands,
                         CRS& crs, BBox& bounds);

/// Read band data into pre-allocated float buffer.
Status read_geotiff_band(const std::string& path,
                         int band_index,    // 0-based
                         float* data,       // host, size = width * height
                         int width, int height);

} // namespace pcr
