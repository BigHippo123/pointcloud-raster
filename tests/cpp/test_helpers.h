#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include "pcr/core/grid.h"
#include "pcr/core/point_cloud.h"
#include <gtest/gtest.h>
#include <cmath>

namespace pcr {
namespace test {

// ---------------------------------------------------------------------------
// Float comparison with tolerance
// ---------------------------------------------------------------------------
inline bool approx_equal(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

inline bool approx_equal(float a, float b, float tol = 1e-6f) {
    return std::abs(a - b) < tol;
}

// ---------------------------------------------------------------------------
// Create a test GridConfig with sensible defaults
// ---------------------------------------------------------------------------
inline GridConfig make_test_grid_config(
    double min_x = 0.0, double min_y = 0.0,
    double max_x = 1000.0, double max_y = 1000.0,
    double cell_size = 1.0)
{
    GridConfig cfg;
    cfg.bounds.min_x = min_x;
    cfg.bounds.min_y = min_y;
    cfg.bounds.max_x = max_x;
    cfg.bounds.max_y = max_y;

    cfg.cell_size_x = cell_size;
    cfg.cell_size_y = -cell_size;  // north-up convention

    cfg.crs = CRS::from_epsg(3857);  // Web Mercator

    cfg.tile_width = 256;
    cfg.tile_height = 256;

    cfg.compute_dimensions();

    return cfg;
}

// ---------------------------------------------------------------------------
// Create a small test grid
// ---------------------------------------------------------------------------
inline std::unique_ptr<Grid> make_test_grid(
    int cols = 10, int rows = 10,
    int num_bands = 1,
    MemoryLocation loc = MemoryLocation::Host)
{
    std::vector<BandDesc> bands;
    for (int i = 0; i < num_bands; ++i) {
        BandDesc desc;
        desc.name = "band" + std::to_string(i);
        desc.dtype = DataType::Float32;
        bands.push_back(desc);
    }

    return Grid::create(cols, rows, bands, loc);
}

// ---------------------------------------------------------------------------
// Create a small test point cloud
// ---------------------------------------------------------------------------
inline std::unique_ptr<PointCloud> make_test_point_cloud(
    size_t capacity = 100,
    MemoryLocation loc = MemoryLocation::Host)
{
    return PointCloud::create(capacity, loc);
}

// ---------------------------------------------------------------------------
// Fill a point cloud with test data
// ---------------------------------------------------------------------------
inline void fill_test_points(PointCloud& cloud, size_t count) {
    if (count > cloud.capacity()) {
        count = cloud.capacity();
    }

    cloud.resize(count);

    double* x = cloud.x();
    double* y = cloud.y();

    for (size_t i = 0; i < count; ++i) {
        x[i] = static_cast<double>(i);
        y[i] = static_cast<double>(i) * 2.0;
    }
}

} // namespace test
} // namespace pcr
