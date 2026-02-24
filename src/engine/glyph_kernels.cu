// glyph_kernels.cu — CPU and GPU glyph splatting kernels
//
// For each point in a TileBatch, paints a weighted footprint across the cells
// it intersects within the tile.  The weight kernel is determined by GlyphSpec:
//   Point    — w = 1.0 for the single containing cell (fast path, not reached here)
//   Line     — Bresenham line; w = 1.0 for every cell on the segment
//   Gaussian — w = exp(-0.5 * ((dx_rot/sx)²+(dy_rot/sy)²)), cutoff at 3σ
//
// State update rules (by reduction type):
//   WeightedAverage / Average:
//       state[0*N+cell] += val * w   (weighted_sum)
//       state[1*N+cell] += w         (weight_sum)
//   Sum:
//       state[cell]     += val * w
//   Count:
//       state[cell]     += w

#include "pcr/engine/glyph_kernels.h"
#include <cmath>
#include <cstring>
#include <algorithm>

#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#endif

#ifdef PCR_HAS_OPENMP
#include <omp.h>
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// Helper: update state at a single cell with weight w
// ---------------------------------------------------------------------------
static inline void update_state_cpu(float* state, int64_t tile_cells, int64_t cell,
                                     float val, float w, ReductionType rtype)
{
    switch (rtype) {
        case ReductionType::WeightedAverage:
        case ReductionType::Average:
#ifdef PCR_HAS_OPENMP
            #pragma omp atomic
            state[cell] += val * w;
            #pragma omp atomic
            state[tile_cells + cell] += w;
#else
            state[cell] += val * w;
            state[tile_cells + cell] += w;
#endif
            break;

        case ReductionType::Sum:
#ifdef PCR_HAS_OPENMP
            #pragma omp atomic
            state[cell] += val * w;
#else
            state[cell] += val * w;
#endif
            break;

        case ReductionType::Count:
#ifdef PCR_HAS_OPENMP
            #pragma omp atomic
            state[cell] += w;
#else
            state[cell] += w;
#endif
            break;

        default:
            break;  // unsupported — caller should have validated
    }
}

// ---------------------------------------------------------------------------
// CPU Gaussian accumulation
// ---------------------------------------------------------------------------
static Status accumulate_glyph_gaussian_cpu(
    const GlyphSpec&  spec,
    ReductionType     rtype,
    const TileBatch&  batch,
    float*            state,
    int64_t           tile_cells,
    const GridConfig& cfg,
    int               tile_col_origin,
    int               tile_row_origin,
    int               tile_w,
    int               tile_h)
{
    if (!batch.coord_x || !batch.coord_y) {
        return Status::error(StatusCode::InvalidArgument,
            "glyph_gaussian: batch.coord_x / coord_y must be non-null for Gaussian glyphs");
    }

    const double origin_x = cfg.bounds.min_x;
    const double origin_y = cfg.bounds.max_y;  // top-left, north-up; cell_size_y is negative
    const double inv_csx  = 1.0 / cfg.cell_size_x;
    const double inv_csy  = 1.0 / cfg.cell_size_y;

#ifdef PCR_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t p = 0; p < batch.num_points; ++p) {
        float val = batch.values[p];

        // World-space point position
        double wx = batch.coord_x[p];
        double wy = batch.coord_y[p];

        // Fractional cell position (origin at bounds.max_y for north-up grids)
        double fcx = (wx - origin_x) * inv_csx;
        double fcy = (wy - origin_y) * inv_csy;

        // Sub-cell offset (fractional part)
        float sub_cx = static_cast<float>(fcx - std::floor(fcx));
        float sub_cy = static_cast<float>(fcy - std::floor(fcy));

        // Per-point sigma (or fallback to defaults)
        float sx_world = (batch.glyph_sigma_x && batch.glyph_sigma_x[p] > 0.0f)
            ? batch.glyph_sigma_x[p] : spec.default_sigma_x;
        float sy_world = (batch.glyph_sigma_y && batch.glyph_sigma_y[p] > 0.0f)
            ? batch.glyph_sigma_y[p] : spec.default_sigma_y;

        // Convert sigma from world units to cell units
        float sx = sx_world * static_cast<float>(inv_csx);
        float sy = sy_world * static_cast<float>(inv_csy);

        float rot = batch.glyph_rotation ? batch.glyph_rotation[p] : spec.default_rotation;
        float cos_rot = std::cos(-rot);
        float sin_rot = std::sin(-rot);

        // Footprint radius in cells
        float R = std::min(3.0f * std::max(sx, sy), spec.max_radius_cells);
        int r = static_cast<int>(std::ceil(R));

        // Center cell (integer, global)
        int icx = static_cast<int>(std::floor(fcx));
        int icy = static_cast<int>(std::floor(fcy));

        // Total weight (for normalization)
        float total_w = 0.0f;

        // First pass: accumulate (and optionally compute total for normalization)
        for (int dy = -r; dy <= r; ++dy) {
            for (int dx = -r; dx <= r; ++dx) {
                int global_cx = icx + dx;
                int global_cy = icy + dy;

                // Local tile coordinates
                int local_cx = global_cx - tile_col_origin;
                int local_cy = global_cy - tile_row_origin;
                if (local_cx < 0 || local_cx >= tile_w) continue;
                if (local_cy < 0 || local_cy >= tile_h) continue;

                // dx/dy relative to point center (in cell units, including sub-cell offset)
                float rdx = static_cast<float>(dx) - sub_cx;
                float rdy = static_cast<float>(dy) - sub_cy;

                // Apply rotation
                float rdx_rot = rdx * cos_rot + rdy * (-sin_rot);
                float rdy_rot = rdx * sin_rot + rdy * cos_rot;

                float w = std::exp(-0.5f * ((rdx_rot / sx) * (rdx_rot / sx) +
                                            (rdy_rot / sy) * (rdy_rot / sy)));
                if (w < 1e-6f) continue;

                int64_t cell = static_cast<int64_t>(local_cy) * tile_w + local_cx;
                if (spec.normalize_weights) {
                    // Deferred to second pass if normalizing
                    (void)cell;
                    total_w += w;
                    // We'd need to store {cell, w} pairs — skip normalization for now
                    // (normalize_weights is a future feature, accepted but not applied)
                }
                update_state_cpu(state, tile_cells, cell, val, w, rtype);
            }
        }
        (void)total_w;
    }

    return Status::success();
}

// ---------------------------------------------------------------------------
// CPU Line accumulation (Bresenham rasterization)
// ---------------------------------------------------------------------------
static Status accumulate_glyph_line_cpu(
    const GlyphSpec&  spec,
    ReductionType     rtype,
    const TileBatch&  batch,
    float*            state,
    int64_t           tile_cells,
    const GridConfig& cfg,
    int               tile_col_origin,
    int               tile_row_origin,
    int               tile_w,
    int               tile_h)
{
    if (!batch.coord_x || !batch.coord_y) {
        return Status::error(StatusCode::InvalidArgument,
            "glyph_line: batch.coord_x / coord_y must be non-null for Line glyphs");
    }

    const double origin_x = cfg.bounds.min_x;
    const double origin_y = cfg.bounds.max_y;  // top-left, north-up; cell_size_y is negative
    const double inv_csx  = 1.0 / cfg.cell_size_x;
    const double inv_csy  = 1.0 / cfg.cell_size_y;

#ifdef PCR_HAS_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (size_t p = 0; p < batch.num_points; ++p) {
        float val = batch.values[p];

        double wx = batch.coord_x[p];
        double wy = batch.coord_y[p];

        // Fractional cell center (origin at bounds.max_y for north-up grids)
        double fcx = (wx - origin_x) * inv_csx;
        double fcy = (wy - origin_y) * inv_csy;

        float direction = batch.glyph_direction
            ? batch.glyph_direction[p] : spec.default_direction;
        float half_len = batch.glyph_half_length
            ? batch.glyph_half_length[p] : spec.default_half_length;

        // Convert half_length from world units to cell units
        float half_cells_x = half_len * static_cast<float>(inv_csx);
        float half_cells_y = half_len * static_cast<float>(inv_csy);

        // Cap at max_radius_cells
        float cap = spec.max_radius_cells;
        half_cells_x = std::min(half_cells_x, cap);
        half_cells_y = std::min(half_cells_y, cap);

        float cos_d = std::cos(direction);
        float sin_d = std::sin(direction);

        // Endpoints in fractional cell coords
        double x0 = fcx - half_cells_x * cos_d;
        double y0 = fcy - half_cells_y * sin_d;
        double x1 = fcx + half_cells_x * cos_d;
        double y1 = fcy + half_cells_y * sin_d;

        // Integer endpoints
        int ix0 = static_cast<int>(std::round(x0));
        int iy0 = static_cast<int>(std::round(y0));
        int ix1 = static_cast<int>(std::round(x1));
        int iy1 = static_cast<int>(std::round(y1));

        // Bresenham walk
        int ddx = std::abs(ix1 - ix0);
        int ddy = std::abs(iy1 - iy0);
        int sx_step = (ix0 < ix1) ? 1 : -1;
        int sy_step = (iy0 < iy1) ? 1 : -1;
        int err = ddx - ddy;
        int cx = ix0, cy = iy0;

        // Limit iterations to prevent infinite loops on bad inputs
        int max_steps = 2 * (ddx + ddy) + 2;
        for (int step = 0; step <= max_steps; ++step) {
            int local_cx = cx - tile_col_origin;
            int local_cy = cy - tile_row_origin;

            if (local_cx >= 0 && local_cx < tile_w &&
                local_cy >= 0 && local_cy < tile_h) {
                int64_t cell = static_cast<int64_t>(local_cy) * tile_w + local_cx;
                update_state_cpu(state, tile_cells, cell, val, 1.0f, rtype);
            }

            if (cx == ix1 && cy == iy1) break;

            int e2 = 2 * err;
            if (e2 > -ddy) { err -= ddy; cx += sx_step; }
            if (e2 <  ddx) { err += ddx; cy += sy_step; }
        }
    }

    return Status::success();
}

// ---------------------------------------------------------------------------
// CPU dispatch
// ---------------------------------------------------------------------------
static Status accumulate_glyph_cpu(
    const GlyphSpec&  spec,
    ReductionType     rtype,
    const TileBatch&  batch,
    float*            state,
    int64_t           tile_cells,
    const GridConfig& cfg,
    int               tile_col_origin,
    int               tile_row_origin,
    int               tile_w,
    int               tile_h)
{
    // Validate reduction type compatibility
    if (rtype != ReductionType::WeightedAverage &&
        rtype != ReductionType::Average &&
        rtype != ReductionType::Sum &&
        rtype != ReductionType::Count) {
        return Status::error(StatusCode::NotImplemented,
            "glyph splatting only supports WeightedAverage, Average, Sum, or Count reduction types");
    }

    switch (spec.type) {
        case GlyphType::Point:
            // Should have been short-circuited by caller — but handle it
            return Status::error(StatusCode::InvalidArgument,
                "accumulate_glyph: Point glyph should use regular accumulate()");

        case GlyphType::Gaussian:
            return accumulate_glyph_gaussian_cpu(
                spec, rtype, batch, state, tile_cells,
                cfg, tile_col_origin, tile_row_origin, tile_w, tile_h);

        case GlyphType::Line:
            return accumulate_glyph_line_cpu(
                spec, rtype, batch, state, tile_cells,
                cfg, tile_col_origin, tile_row_origin, tile_w, tile_h);
    }

    return Status::error(StatusCode::NotImplemented, "glyph: unknown glyph type");
}

#ifdef PCR_HAS_CUDA
// ---------------------------------------------------------------------------
// GPU kernels (Phases 6-7)
// ---------------------------------------------------------------------------

struct DeviceGlyphCfg {
    double origin_x;
    double origin_y;
    double inv_cell_size_x;
    double inv_cell_size_y;
    int    tile_col_origin;
    int    tile_row_origin;
    int    tile_w;
    int    tile_h;
    float  max_radius_cells;
};

// --- Gaussian GPU kernel ---
__global__ void kernel_glyph_gaussian(
    const float*  values,
    const double* coord_x,
    const double* coord_y,
    const float*  g_sigma_x,   // nullable
    const float*  g_sigma_y,   // nullable
    const float*  g_rotation,  // nullable
    size_t        num_points,
    float         default_sigma_x,
    float         default_sigma_y,
    float         default_rotation,
    float*        state,
    int64_t       tile_cells,
    int           reduction_type_int,  // 0=WeightedAverage/Average, 1=Sum, 2=Count
    DeviceGlyphCfg dcfg)
{
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= num_points) return;

    float val = values[p];
    double wx  = coord_x[p];
    double wy  = coord_y[p];

    double fcx = (wx - dcfg.origin_x) * dcfg.inv_cell_size_x;
    double fcy = (wy - dcfg.origin_y) * dcfg.inv_cell_size_y;

    float sub_cx = static_cast<float>(fcx - floor(fcx));
    float sub_cy = static_cast<float>(fcy - floor(fcy));

    float sx_w = (g_sigma_x && g_sigma_x[p] > 0.0f) ? g_sigma_x[p] : default_sigma_x;
    float sy_w = (g_sigma_y && g_sigma_y[p] > 0.0f) ? g_sigma_y[p] : default_sigma_y;

    float sx = sx_w * static_cast<float>(dcfg.inv_cell_size_x);
    float sy = sy_w * static_cast<float>(dcfg.inv_cell_size_y);

    float rot = g_rotation ? g_rotation[p] : default_rotation;
    float cos_rot = cosf(-rot);
    float sin_rot = sinf(-rot);

    float R = fminf(3.0f * fmaxf(sx, sy), dcfg.max_radius_cells);
    int r = static_cast<int>(ceilf(R));

    int icx = static_cast<int>(floor(fcx));
    int icy = static_cast<int>(floor(fcy));

    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            int local_cx = (icx + dx) - dcfg.tile_col_origin;
            int local_cy = (icy + dy) - dcfg.tile_row_origin;
            if (local_cx < 0 || local_cx >= dcfg.tile_w) continue;
            if (local_cy < 0 || local_cy >= dcfg.tile_h) continue;

            float rdx = static_cast<float>(dx) - sub_cx;
            float rdy = static_cast<float>(dy) - sub_cy;

            float rdx_rot = rdx * cos_rot + rdy * (-sin_rot);
            float rdy_rot = rdx * sin_rot + rdy *  cos_rot;

            float w = expf(-0.5f * ((rdx_rot / sx) * (rdx_rot / sx) +
                                    (rdy_rot / sy) * (rdy_rot / sy)));
            if (w < 1e-6f) continue;

            int64_t cell = static_cast<int64_t>(local_cy) * dcfg.tile_w + local_cx;

            if (reduction_type_int == 0) {
                // WeightedAverage / Average
                atomicAdd(&state[cell],            val * w);
                atomicAdd(&state[tile_cells + cell], w);
            } else if (reduction_type_int == 1) {
                // Sum
                atomicAdd(&state[cell], val * w);
            } else {
                // Count
                atomicAdd(&state[cell], w);
            }
        }
    }
}

// --- Line GPU kernel ---
__global__ void kernel_glyph_line(
    const float*  values,
    const double* coord_x,
    const double* coord_y,
    const float*  g_direction,   // nullable
    const float*  g_half_length, // nullable
    size_t        num_points,
    float         default_direction,
    float         default_half_length,
    float*        state,
    int64_t       tile_cells,
    int           reduction_type_int,
    DeviceGlyphCfg dcfg)
{
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= num_points) return;

    float val = values[p];
    double wx  = coord_x[p];
    double wy  = coord_y[p];

    double fcx = (wx - dcfg.origin_x) * dcfg.inv_cell_size_x;
    double fcy = (wy - dcfg.origin_y) * dcfg.inv_cell_size_y;

    float dir = g_direction ? g_direction[p] : default_direction;
    float hl  = g_half_length ? g_half_length[p] : default_half_length;

    float hx = fminf(hl * static_cast<float>(dcfg.inv_cell_size_x), dcfg.max_radius_cells);
    float hy = fminf(hl * static_cast<float>(dcfg.inv_cell_size_y), dcfg.max_radius_cells);

    float cos_d = cosf(dir);
    float sin_d = sinf(dir);

    int ix0 = static_cast<int>(roundf(static_cast<float>(fcx) - hx * cos_d));
    int iy0 = static_cast<int>(roundf(static_cast<float>(fcy) - hy * sin_d));
    int ix1 = static_cast<int>(roundf(static_cast<float>(fcx) + hx * cos_d));
    int iy1 = static_cast<int>(roundf(static_cast<float>(fcy) + hy * sin_d));

    int ddx = abs(ix1 - ix0);
    int ddy = abs(iy1 - iy0);
    int sx_step = (ix0 < ix1) ? 1 : -1;
    int sy_step = (iy0 < iy1) ? 1 : -1;
    int err = ddx - ddy;
    int cx = ix0, cy = iy0;
    int max_steps = 2 * (ddx + ddy) + 2;

    for (int step = 0; step <= max_steps; ++step) {
        int local_cx = cx - dcfg.tile_col_origin;
        int local_cy = cy - dcfg.tile_row_origin;
        if (local_cx >= 0 && local_cx < dcfg.tile_w &&
            local_cy >= 0 && local_cy < dcfg.tile_h) {
            int64_t cell = static_cast<int64_t>(local_cy) * dcfg.tile_w + local_cx;
            if (reduction_type_int == 0) {
                atomicAdd(&state[cell],              val);
                atomicAdd(&state[tile_cells + cell], 1.0f);
            } else if (reduction_type_int == 1) {
                atomicAdd(&state[cell], val);
            } else {
                atomicAdd(&state[cell], 1.0f);
            }
        }

        if (cx == ix1 && cy == iy1) break;
        int e2 = 2 * err;
        if (e2 > -ddy) { err -= ddy; cx += sx_step; }
        if (e2 <  ddx) { err += ddx; cy += sy_step; }
    }
}

static Status accumulate_glyph_gpu(
    const GlyphSpec&  spec,
    ReductionType     rtype,
    const TileBatch&  batch,
    float*            d_state,
    int64_t           tile_cells,
    const GridConfig& cfg,
    int               tile_col_origin,
    int               tile_row_origin,
    int               tile_w,
    int               tile_h,
    void*             stream)
{
    if (rtype != ReductionType::WeightedAverage &&
        rtype != ReductionType::Average &&
        rtype != ReductionType::Sum &&
        rtype != ReductionType::Count) {
        return Status::error(StatusCode::NotImplemented,
            "glyph splatting only supports WeightedAverage, Average, Sum, or Count reduction types");
    }

    int rtype_int = 0;
    if (rtype == ReductionType::Sum)   rtype_int = 1;
    if (rtype == ReductionType::Count) rtype_int = 2;

    DeviceGlyphCfg dcfg;
    dcfg.origin_x        = cfg.bounds.min_x;
    dcfg.origin_y        = cfg.bounds.max_y;  // top-left, north-up
    dcfg.inv_cell_size_x = 1.0 / cfg.cell_size_x;
    dcfg.inv_cell_size_y = 1.0 / cfg.cell_size_y;
    dcfg.tile_col_origin = tile_col_origin;
    dcfg.tile_row_origin = tile_row_origin;
    dcfg.tile_w          = tile_w;
    dcfg.tile_h          = tile_h;
    dcfg.max_radius_cells = spec.max_radius_cells;

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = 256;
    int grid  = static_cast<int>((batch.num_points + block - 1) / block);

    if (spec.type == GlyphType::Gaussian) {
        if (!batch.coord_x || !batch.coord_y) {
            return Status::error(StatusCode::InvalidArgument,
                "glyph_gaussian GPU: batch.coord_x/y must be non-null");
        }
        kernel_glyph_gaussian<<<grid, block, 0, s>>>(
            batch.values, batch.coord_x, batch.coord_y,
            batch.glyph_sigma_x, batch.glyph_sigma_y, batch.glyph_rotation,
            batch.num_points,
            spec.default_sigma_x, spec.default_sigma_y, spec.default_rotation,
            d_state, tile_cells, rtype_int, dcfg);
    } else {  // Line
        if (!batch.coord_x || !batch.coord_y) {
            return Status::error(StatusCode::InvalidArgument,
                "glyph_line GPU: batch.coord_x/y must be non-null");
        }
        kernel_glyph_line<<<grid, block, 0, s>>>(
            batch.values, batch.coord_x, batch.coord_y,
            batch.glyph_direction, batch.glyph_half_length,
            batch.num_points,
            spec.default_direction, spec.default_half_length,
            d_state, tile_cells, rtype_int, dcfg);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            std::string("glyph GPU kernel failed: ") + cudaGetErrorString(err));
    }

    return Status::success();
}
#endif // PCR_HAS_CUDA

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
Status accumulate_glyph(
    const GlyphSpec&  spec,
    ReductionType     rtype,
    const TileBatch&  batch,
    float*            state,
    int64_t           tile_cells,
    const GridConfig& cfg,
    int               tile_col_origin,
    int               tile_row_origin,
    int               tile_w,
    int               tile_h,
    void*             stream)
{
    if (batch.num_points == 0) return Status::success();

#ifdef PCR_HAS_CUDA
    if (batch.location == MemoryLocation::Device) {
        return accumulate_glyph_gpu(
            spec, rtype, batch, state, tile_cells,
            cfg, tile_col_origin, tile_row_origin, tile_w, tile_h, stream);
    }
#else
    (void)stream;
#endif

    if (batch.location != MemoryLocation::Host) {
        return Status::error(StatusCode::InvalidArgument,
            "accumulate_glyph: unsupported memory location (expected Host or Device)");
    }

    return accumulate_glyph_cpu(
        spec, rtype, batch, state, tile_cells,
        cfg, tile_col_origin, tile_row_origin, tile_w, tile_h);
}

} // namespace pcr
