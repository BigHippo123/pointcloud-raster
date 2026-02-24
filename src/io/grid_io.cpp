#include "pcr/io/grid_io.h"
#include "pcr/core/grid.h"
#include <gdal.h>
#include <gdal_priv.h>
#include <ogr_spatialref.h>
#include <cpl_conv.h>
#include <sstream>
#include <cstring>

namespace pcr {

// ===========================================================================
// GDAL initialization helper
// ===========================================================================

namespace {

struct GDALInit {
    GDALInit() {
        GDALAllRegister();
    }
};

static GDALInit g_gdal_init;

// Helper to convert StatusCode to string
std::string gdal_error_message(const std::string& context) {
    std::ostringstream oss;
    oss << context << ": " << CPLGetLastErrorMsg();
    return oss.str();
}

} // anonymous namespace

// ===========================================================================
// write_geotiff — single-shot write
// ===========================================================================

Status write_geotiff(
    const std::string& path,
    const Grid& grid,
    const GridConfig& config,
    const GeoTiffOptions& options)
{
    if (grid.location() != MemoryLocation::Host) {
        return Status::error(StatusCode::InvalidArgument, "grid must be on host");
    }

    if (grid.cols() != config.width || grid.rows() != config.height) {
        return Status::error(StatusCode::InvalidArgument, "grid dimensions mismatch config");
    }

    // Get GDAL driver
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) {
        return Status::error(StatusCode::IoError, "GTiff driver not available");
    }

    // Build creation options
    std::vector<std::string> opt_strings;
    std::vector<const char*> opt_ptrs;

    if (options.bigtiff) {
        opt_strings.push_back("BIGTIFF=YES");
    }

    if (!options.compress.empty() && options.compress != "NONE") {
        opt_strings.push_back("COMPRESS=" + options.compress);
        if (options.compress == "DEFLATE" || options.compress == "ZSTD") {
            opt_strings.push_back("ZLEVEL=" + std::to_string(options.compress_level));
        }
    }

    if (options.tile_width > 0 && options.tile_height > 0) {
        opt_strings.push_back("TILED=YES");
        opt_strings.push_back("BLOCKXSIZE=" + std::to_string(options.tile_width));
        opt_strings.push_back("BLOCKYSIZE=" + std::to_string(options.tile_height));
    }

    for (const auto& s : opt_strings) {
        opt_ptrs.push_back(s.c_str());
    }
    opt_ptrs.push_back(nullptr);

    // Create dataset
    GDALDataset* ds = driver->Create(
        path.c_str(),
        config.width,
        config.height,
        grid.num_bands(),
        GDT_Float32,
        const_cast<char**>(opt_ptrs.data()));

    if (!ds) {
        return Status::error(StatusCode::IoError, gdal_error_message("failed to create GeoTIFF"));
    }

    // Set geotransform
    double gt[6];
    config.gdal_geotransform(gt);
    if (ds->SetGeoTransform(gt) != CE_None) {
        GDALClose(ds);
        return Status::error(StatusCode::IoError, "failed to set geotransform");
    }

    // Set CRS
    if (config.crs.is_valid()) {
        OGRSpatialReference srs;
        if (!config.crs.wkt.empty()) {
            OGRErr err = srs.importFromWkt(config.crs.wkt.c_str());
            if (err == OGRERR_NONE) {
                char* wkt_out = nullptr;
                srs.exportToWkt(&wkt_out);
                if (wkt_out) {
                    ds->SetProjection(wkt_out);
                    CPLFree(wkt_out);
                }
            }
        }
    }

    // Write bands
    for (int b = 0; b < grid.num_bands(); ++b) {
        GDALRasterBand* band = ds->GetRasterBand(b + 1);  // GDAL is 1-indexed
        if (!band) {
            GDALClose(ds);
            return Status::error(StatusCode::IoError, "failed to get band " + std::to_string(b));
        }

        // Set band description (name)
        BandDesc desc = grid.band_desc(b);
        band->SetDescription(desc.name.c_str());

        // Set NoData value
        band->SetNoDataValue(NAN);

        // Write raster data
        const float* data = grid.band_f32(b);
        CPLErr err = band->RasterIO(
            GF_Write,
            0, 0,                           // xoff, yoff
            config.width, config.height,    // xsize, ysize
            const_cast<float*>(data),       // buffer
            config.width, config.height,    // buf_xsize, buf_ysize
            GDT_Float32,
            0, 0);                          // pixel_space, line_space

        if (err != CE_None) {
            GDALClose(ds);
            return Status::error(StatusCode::IoError,
                gdal_error_message("failed to write band " + std::to_string(b)));
        }
    }

    // Build overviews if COG requested
    if (options.cloud_optimized) {
        std::vector<int> overview_levels;
        int min_dim = std::min(config.width, config.height);
        for (int level = 2; min_dim / level >= 256; level *= 2) {
            overview_levels.push_back(level);
        }

        if (!overview_levels.empty()) {
            CPLErr err = ds->BuildOverviews(
                options.overview_resampling.c_str(),
                overview_levels.size(),
                overview_levels.data(),
                0, nullptr, // all bands
                nullptr, nullptr);

            if (err != CE_None) {
                // Non-fatal, just warn
                CPLError(CE_Warning, CPLE_AppDefined,
                    "Failed to build overviews, file will not be fully COG-compliant");
            }
        }
    }

    // Close
    GDALClose(ds);
    return Status::success();
}

// ===========================================================================
// TiledGeoTiffWriter — incremental tile writing
// ===========================================================================

struct TiledGeoTiffWriter::Impl {
    GDALDataset* ds = nullptr;
    GridConfig config;
    int num_bands = 0;
    std::vector<std::string> band_names;
    GeoTiffOptions options;
};

TiledGeoTiffWriter::~TiledGeoTiffWriter() {
    if (impl_ && impl_->ds) {
        GDALClose(impl_->ds);
    }
}

std::unique_ptr<TiledGeoTiffWriter> TiledGeoTiffWriter::open(
    const std::string& path,
    const GridConfig& config,
    const std::vector<std::string>& band_names,
    const GeoTiffOptions& options)
{
    if (band_names.empty()) {
        return nullptr;
    }

    Status s = config.validate();
    if (!s.ok()) {
        return nullptr;
    }

    // Get GDAL driver
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    if (!driver) {
        return nullptr;
    }

    // Build creation options
    std::vector<std::string> opt_strings;
    std::vector<const char*> opt_ptrs;

    if (options.bigtiff) {
        opt_strings.push_back("BIGTIFF=YES");
    }

    if (!options.compress.empty() && options.compress != "NONE") {
        opt_strings.push_back("COMPRESS=" + options.compress);
        if (options.compress == "DEFLATE" || options.compress == "ZSTD") {
            opt_strings.push_back("ZLEVEL=" + std::to_string(options.compress_level));
        }
    }

    if (options.tile_width > 0 && options.tile_height > 0) {
        opt_strings.push_back("TILED=YES");
        opt_strings.push_back("BLOCKXSIZE=" + std::to_string(options.tile_width));
        opt_strings.push_back("BLOCKYSIZE=" + std::to_string(options.tile_height));
    }

    for (const auto& s : opt_strings) {
        opt_ptrs.push_back(s.c_str());
    }
    opt_ptrs.push_back(nullptr);

    // Create dataset
    GDALDataset* ds = driver->Create(
        path.c_str(),
        config.width,
        config.height,
        band_names.size(),
        GDT_Float32,
        const_cast<char**>(opt_ptrs.data()));

    if (!ds) {
        return nullptr;
    }

    // Set geotransform
    double gt[6];
    config.gdal_geotransform(gt);
    ds->SetGeoTransform(gt);

    // Set CRS
    if (config.crs.is_valid()) {
        OGRSpatialReference srs;
        if (!config.crs.wkt.empty()) {
            OGRErr err = srs.importFromWkt(config.crs.wkt.c_str());
            if (err == OGRERR_NONE) {
                char* wkt_out = nullptr;
                srs.exportToWkt(&wkt_out);
                if (wkt_out) {
                    ds->SetProjection(wkt_out);
                    CPLFree(wkt_out);
                }
            }
        }
    }

    // Set band descriptions and NoData
    for (size_t i = 0; i < band_names.size(); ++i) {
        GDALRasterBand* band = ds->GetRasterBand(i + 1);
        if (band) {
            band->SetDescription(band_names[i].c_str());
            band->SetNoDataValue(NAN);
        }
    }

    // Create writer object
    auto writer = std::unique_ptr<TiledGeoTiffWriter>(new TiledGeoTiffWriter());
    writer->impl_ = std::make_unique<Impl>();
    writer->impl_->ds = ds;
    writer->impl_->config = config;
    writer->impl_->num_bands = band_names.size();
    writer->impl_->band_names = band_names;
    writer->impl_->options = options;

    return writer;
}

Status TiledGeoTiffWriter::write_tile(TileIndex tile, const float* data, int num_bands) {
    if (!impl_ || !impl_->ds) {
        return Status::error(StatusCode::InvalidArgument, "writer not open");
    }

    if (num_bands != impl_->num_bands) {
        return Status::error(StatusCode::InvalidArgument, "band count mismatch");
    }

    if (!data) {
        return Status::error(StatusCode::InvalidArgument, "null data pointer");
    }

    // Get tile cell range
    int col_start, row_start, col_count, row_count;
    impl_->config.tile_cell_range(tile, col_start, row_start, col_count, row_count);

    // Write each band
    for (int b = 0; b < num_bands; ++b) {
        GDALRasterBand* band = impl_->ds->GetRasterBand(b + 1);
        if (!band) {
            return Status::error(StatusCode::IoError, "failed to get band " + std::to_string(b));
        }

        // Data is band-sequential: band b starts at offset b * tile_cells
        const int64_t tile_cells = col_count * row_count;
        const float* band_data = data + b * tile_cells;

        CPLErr err = band->RasterIO(
            GF_Write,
            col_start, row_start,       // xoff, yoff
            col_count, row_count,       // xsize, ysize
            const_cast<float*>(band_data),  // buffer
            col_count, row_count,       // buf_xsize, buf_ysize
            GDT_Float32,
            0, 0);                      // pixel_space, line_space

        if (err != CE_None) {
            return Status::error(StatusCode::IoError,
                gdal_error_message("failed to write tile data for band " + std::to_string(b)));
        }
    }

    return Status::success();
}

Status TiledGeoTiffWriter::close() {
    if (!impl_ || !impl_->ds) {
        return Status::error(StatusCode::InvalidArgument, "writer not open");
    }

    // Build overviews if COG requested
    if (impl_->options.cloud_optimized) {
        std::vector<int> overview_levels;
        int min_dim = std::min(impl_->config.width, impl_->config.height);
        for (int level = 2; min_dim / level >= 256; level *= 2) {
            overview_levels.push_back(level);
        }

        if (!overview_levels.empty()) {
            CPLErr err = impl_->ds->BuildOverviews(
                impl_->options.overview_resampling.c_str(),
                overview_levels.size(),
                overview_levels.data(),
                0, nullptr,
                nullptr, nullptr);

            if (err != CE_None) {
                CPLError(CE_Warning, CPLE_AppDefined,
                    "Failed to build overviews, file will not be fully COG-compliant");
            }
        }
    }

    // Close dataset
    GDALClose(impl_->ds);
    impl_->ds = nullptr;

    return Status::success();
}

// ===========================================================================
// GeoTIFF reader functions
// ===========================================================================

Status read_geotiff_info(
    const std::string& path,
    int& width, int& height, int& num_bands,
    CRS& crs, BBox& bounds)
{
    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpen(path.c_str(), GA_ReadOnly));
    if (!ds) {
        return Status::error(StatusCode::IoError, gdal_error_message("failed to open file: " + path));
    }

    width = ds->GetRasterXSize();
    height = ds->GetRasterYSize();
    num_bands = ds->GetRasterCount();

    // Get geotransform
    double gt[6];
    if (ds->GetGeoTransform(gt) == CE_None) {
        // gt[0] = origin_x
        // gt[1] = pixel_width
        // gt[2] = rotation (usually 0)
        // gt[3] = origin_y
        // gt[4] = rotation (usually 0)
        // gt[5] = pixel_height (negative for north-up)

        double min_x = gt[0];
        double max_y = gt[3];
        double max_x = min_x + gt[1] * width;
        double min_y = max_y + gt[5] * height;

        bounds.min_x = min_x;
        bounds.max_x = max_x;
        bounds.min_y = min_y;
        bounds.max_y = max_y;
    }

    // Get CRS
    const char* wkt_cstr = ds->GetProjectionRef();
    if (wkt_cstr && strlen(wkt_cstr) > 0) {
        crs.wkt = wkt_cstr;

        // Try to extract EPSG code
        OGRSpatialReference srs;
        if (srs.importFromWkt(wkt_cstr) == OGRERR_NONE) {
            const char* authority = srs.GetAuthorityName(nullptr);
            const char* code = srs.GetAuthorityCode(nullptr);
            if (authority && code && strcmp(authority, "EPSG") == 0) {
                crs.epsg = atoi(code);
            }
        }
    }

    GDALClose(ds);
    return Status::success();
}

Status read_geotiff_band(
    const std::string& path,
    int band_index,
    float* data,
    int width, int height)
{
    if (!data) {
        return Status::error(StatusCode::InvalidArgument, "null data pointer");
    }

    if (band_index < 0) {
        return Status::error(StatusCode::InvalidArgument, "invalid band index");
    }

    GDALDataset* ds = static_cast<GDALDataset*>(GDALOpen(path.c_str(), GA_ReadOnly));
    if (!ds) {
        return Status::error(StatusCode::IoError, gdal_error_message("failed to open file: " + path));
    }

    if (ds->GetRasterXSize() != width || ds->GetRasterYSize() != height) {
        GDALClose(ds);
        return Status::error(StatusCode::InvalidArgument, "dimension mismatch");
    }

    if (band_index >= ds->GetRasterCount()) {
        GDALClose(ds);
        return Status::error(StatusCode::InvalidArgument, "band index out of range");
    }

    GDALRasterBand* band = ds->GetRasterBand(band_index + 1);  // GDAL is 1-indexed
    if (!band) {
        GDALClose(ds);
        return Status::error(StatusCode::IoError, "failed to get band");
    }

    CPLErr err = band->RasterIO(
        GF_Read,
        0, 0,               // xoff, yoff
        width, height,      // xsize, ysize
        data,               // buffer
        width, height,      // buf_xsize, buf_ysize
        GDT_Float32,
        0, 0);              // pixel_space, line_space

    if (err != CE_None) {
        GDALClose(ds);
        return Status::error(StatusCode::IoError, gdal_error_message("failed to read band data"));
    }

    GDALClose(ds);
    return Status::success();
}

} // namespace pcr
