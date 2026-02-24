#include "pcr/engine/tile_router.h"
#include "pcr/core/point_cloud.h"
#include "pcr/engine/memory_pool.h"
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>

#ifdef PCR_HAS_CUDA
#include "pcr/engine/tile_router_kernels.h"
#include <cuda_runtime.h>
#endif

#ifdef PCR_HAS_OPENMP
#include <omp.h>
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// TileRouter implementation
// ---------------------------------------------------------------------------
struct TileRouter::Impl {
    GridConfig config;
    MemoryPool* pool;

    // Temporary storage for sorted data (CPU only)
    std::vector<uint32_t> sort_keys;     // Combined (tile_index << 32) | cell_index for sorting
    std::vector<size_t>   sort_indices;  // Original point indices

    Impl(const GridConfig& cfg, MemoryPool* p)
        : config(cfg), pool(p) {}
};

TileRouter::~TileRouter() = default;

std::unique_ptr<TileRouter> TileRouter::create(const GridConfig& config, MemoryPool* pool) {
    auto router = std::unique_ptr<TileRouter>(new TileRouter);
    router->impl_ = std::make_unique<Impl>(config, pool);
    return router;
}

// ---------------------------------------------------------------------------
// CPU implementation (always available)
// ---------------------------------------------------------------------------
// These functions are available in both CPU-only and GPU builds.
// In GPU builds, they provide a fallback for Host memory data.
// ---------------------------------------------------------------------------

/// CPU implementation of assign(): convert points to tile/cell indices
static Status assign_cpu(const PointCloud& cloud, TileAssignment& out,
                         const GridConfig& config, void* /*stream*/) {
    // Validate
    if (cloud.location() != MemoryLocation::Host) {
        return Status::error(StatusCode::InvalidArgument,
            "CPU tile router requires Host memory location");
    }

    size_t num_points = cloud.count();
    if (num_points == 0) {
        out.cell_indices = nullptr;
        out.tile_indices = nullptr;
        out.valid_mask = nullptr;
        out.num_points = 0;
        out.location = MemoryLocation::Host;
        return Status::success();
    }

    // Allocate output arrays
    out.cell_indices = static_cast<uint32_t*>(malloc(num_points * sizeof(uint32_t)));
    out.tile_indices = static_cast<uint32_t*>(malloc(num_points * sizeof(uint32_t)));
    out.valid_mask = static_cast<uint8_t*>(malloc(num_points * sizeof(uint8_t)));
    out.num_points = num_points;

    if (!out.cell_indices || !out.tile_indices || !out.valid_mask) {
        free(out.cell_indices);
        free(out.tile_indices);
        free(out.valid_mask);
        return Status::error(StatusCode::InvalidArgument,
            "tile router: failed to allocate output arrays");
    }

    // Get point coordinates
    const double* x = cloud.x();
    const double* y = cloud.y();

    // Compute tile and cell indices for each point
    // This loop is perfectly parallel - each point is independent
#ifdef PCR_HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (size_t i = 0; i < num_points; ++i) {
        double px = x[i];
        double py = y[i];

        // Convert world coordinates to grid cell
        int cx, cy;
        bool in_bounds = config.world_to_cell(px, py, cx, cy);

        // Check if point is within grid bounds
        if (!in_bounds) {
            out.valid_mask[i] = 0;
            out.cell_indices[i] = 0;
            out.tile_indices[i] = 0;
            continue;
        }

        out.valid_mask[i] = 1;

        // Compute global cell index (row-major)
        uint32_t cell_idx = static_cast<uint32_t>(cy * config.width + cx);
        out.cell_indices[i] = cell_idx;

        // Compute tile indices
        int tile_col = cx / config.tile_width;
        int tile_row = cy / config.tile_height;

        // Pack tile index as (tile_row * tiles_x + tile_col)
        int tiles_x = (config.width + config.tile_width - 1) / config.tile_width;
        uint32_t tile_idx = static_cast<uint32_t>(tile_row * tiles_x + tile_col);
        out.tile_indices[i] = tile_idx;
    }

    out.location = MemoryLocation::Host;
    return Status::success();
}

/// CPU implementation of sort(): sort points by tile and cell indices
/// Uses std::sort with a custom comparator. Sorts invalid points to the end.
/// @param assignment TileAssignment to sort (modified in-place)
/// @param values Optional value array to sort alongside (can be nullptr)
/// @param weights Optional weight array to sort alongside (can be nullptr)
/// @param timestamps Optional timestamp array to sort alongside (can be nullptr)
/// @param glyph Optional glyph arrays to co-sort (can be nullptr)
/// @param config Grid configuration (unused, kept for signature consistency)
/// @param sort_indices_buffer Working buffer for sort indices
/// @param stream CUDA stream (unused in CPU implementation)
static Status sort_cpu(TileAssignment& assignment,
                       float* values,
                       float* weights,
                       float* timestamps,
                       GlyphSortArrays* glyph,
                       const GridConfig& /*config*/,
                       std::vector<size_t>& sort_indices_buffer,
                       void* /*stream*/) {
    size_t num_points = assignment.num_points;
    if (num_points == 0) {
        return Status::success();
    }

    // Create sort indices
    sort_indices_buffer.resize(num_points);
    for (size_t i = 0; i < num_points; ++i) {
        sort_indices_buffer[i] = i;
    }

    // Sort by (tile_index, cell_index)
    // We sort indices array based on tile and cell indices
    std::sort(sort_indices_buffer.begin(), sort_indices_buffer.end(),
        [&assignment](size_t a, size_t b) {
            // Skip invalid points (sort them to end)
            if (!assignment.valid_mask[a]) return false;
            if (!assignment.valid_mask[b]) return true;

            // Sort by tile first, then by cell within tile
            uint32_t tile_a = assignment.tile_indices[a];
            uint32_t tile_b = assignment.tile_indices[b];
            if (tile_a != tile_b) {
                return tile_a < tile_b;
            }
            return assignment.cell_indices[a] < assignment.cell_indices[b];
        });

    // Apply permutation to cell_indices, tile_indices, and value arrays
    std::vector<uint32_t> temp_cells(num_points);
    std::vector<uint32_t> temp_tiles(num_points);
    std::vector<uint8_t> temp_valid(num_points);

    for (size_t i = 0; i < num_points; ++i) {
        size_t src = sort_indices_buffer[i];
        temp_cells[i] = assignment.cell_indices[src];
        temp_tiles[i] = assignment.tile_indices[src];
        temp_valid[i] = assignment.valid_mask[src];
    }

    memcpy(assignment.cell_indices, temp_cells.data(), num_points * sizeof(uint32_t));
    memcpy(assignment.tile_indices, temp_tiles.data(), num_points * sizeof(uint32_t));
    memcpy(assignment.valid_mask, temp_valid.data(), num_points * sizeof(uint8_t));

    // Sort value arrays if provided
    if (values) {
        std::vector<float> temp_values(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            temp_values[i] = values[sort_indices_buffer[i]];
        }
        memcpy(values, temp_values.data(), num_points * sizeof(float));
    }

    if (weights) {
        std::vector<float> temp_weights(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            temp_weights[i] = weights[sort_indices_buffer[i]];
        }
        memcpy(weights, temp_weights.data(), num_points * sizeof(float));
    }

    if (timestamps) {
        std::vector<float> temp_timestamps(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            temp_timestamps[i] = timestamps[sort_indices_buffer[i]];
        }
        memcpy(timestamps, temp_timestamps.data(), num_points * sizeof(float));
    }

    // Co-sort glyph arrays
    if (glyph) {
        auto apply_perm_f32 = [&](float* arr) {
            if (!arr) return;
            std::vector<float> tmp(num_points);
            for (size_t i = 0; i < num_points; ++i) tmp[i] = arr[sort_indices_buffer[i]];
            memcpy(arr, tmp.data(), num_points * sizeof(float));
        };
        auto apply_perm_f64 = [&](double* arr) {
            if (!arr) return;
            std::vector<double> tmp(num_points);
            for (size_t i = 0; i < num_points; ++i) tmp[i] = arr[sort_indices_buffer[i]];
            memcpy(arr, tmp.data(), num_points * sizeof(double));
        };

        apply_perm_f64(glyph->coord_x);
        apply_perm_f64(glyph->coord_y);
        apply_perm_f32(glyph->direction);
        apply_perm_f32(glyph->half_length);
        apply_perm_f32(glyph->sigma_x);
        apply_perm_f32(glyph->sigma_y);
        apply_perm_f32(glyph->rotation);
    }

    return Status::success();
}

/// CPU implementation of extract_batches(): split sorted points into per-tile batches
/// Scans through sorted points and creates TileBatch objects for each tile.
/// Converts global cell indices to local (within-tile) indices.
/// Allocates new arrays for local_cell_indices (caller must free).
/// @param assignment Sorted TileAssignment (must be sorted by tile first)
/// @param sorted_values Optional sorted value array (can be nullptr)
/// @param sorted_weights Optional sorted weight array (can be nullptr)
/// @param sorted_timestamps Optional sorted timestamp array (can be nullptr)
/// @param out_batches Output vector of TileBatch objects (cleared first)
/// @param config Grid configuration for tile layout
/// @param glyph Optional sorted glyph arrays — pointers propagated into batches
static Status extract_batches_cpu(const TileAssignment& assignment,
                                   float* sorted_values,
                                   float* sorted_weights,
                                   float* sorted_timestamps,
                                   std::vector<TileBatch>& out_batches,
                                   const GridConfig& config,
                                   GlyphSortArrays* glyph = nullptr) {
    out_batches.clear();

    size_t num_points = assignment.num_points;
    if (num_points == 0) {
        return Status::success();
    }

    // Scan through sorted points and split into per-tile batches
    size_t batch_start = 0;
    uint32_t current_tile = assignment.tile_indices[0];

    // Skip invalid points at the beginning
    while (batch_start < num_points && !assignment.valid_mask[batch_start]) {
        ++batch_start;
    }

    if (batch_start >= num_points) {
        return Status::success();  // All points invalid
    }

    current_tile = assignment.tile_indices[batch_start];

    for (size_t i = batch_start + 1; i <= num_points; ++i) {
        // Check if we've reached a new tile or end of array
        bool new_batch = (i == num_points) ||
                        (!assignment.valid_mask[i]) ||
                        (assignment.tile_indices[i] != current_tile);

        if (new_batch) {
            size_t batch_size = i - batch_start;

            // Decode tile index to (row, col)
            int tiles_x = (config.width + config.tile_width - 1) / config.tile_width;
            int tile_row = current_tile / tiles_x;
            int tile_col = current_tile % tiles_x;

            // Get tile origin in grid coordinates
            int tile_x0 = tile_col * config.tile_width;
            int tile_y0 = tile_row * config.tile_height;

            // Calculate actual tile dimensions (may be smaller than configured for edge tiles)
            int actual_tile_width = std::min(config.tile_width, config.width - tile_x0);
            int actual_tile_height = std::min(config.tile_height, config.height - tile_y0);

            // Convert global cell indices to local (within-tile) indices
            // Note: We need to allocate new arrays for local indices
            uint32_t* local_cells = static_cast<uint32_t*>(malloc(batch_size * sizeof(uint32_t)));
            if (!local_cells) {
                return Status::error(StatusCode::InvalidArgument,
                    "extract_batches: failed to allocate local cell indices");
            }

            for (size_t j = 0; j < batch_size; ++j) {
                uint32_t global_cell = assignment.cell_indices[batch_start + j];
                int cx = global_cell % config.width;
                int cy = global_cell / config.width;

                // Convert to local tile coordinates
                int local_cx = cx - tile_x0;
                int local_cy = cy - tile_y0;

                // Local cell index within tile (use actual tile width, not configured width)
                local_cells[j] = static_cast<uint32_t>(local_cy * actual_tile_width + local_cx);
            }

            // Create batch
            TileBatch batch;
            batch.tile = {tile_row, tile_col};
            batch.local_cell_indices = local_cells;
            batch.values = sorted_values ? (sorted_values + batch_start) : nullptr;
            batch.weights = sorted_weights ? (sorted_weights + batch_start) : nullptr;
            batch.timestamps = sorted_timestamps ? (sorted_timestamps + batch_start) : nullptr;
            batch.num_points = batch_size;
            batch.location = MemoryLocation::Host;

            // Propagate glyph array pointers (offset into sorted arrays)
            if (glyph) {
                batch.coord_x           = glyph->coord_x      ? glyph->coord_x      + batch_start : nullptr;
                batch.coord_y           = glyph->coord_y      ? glyph->coord_y      + batch_start : nullptr;
                batch.glyph_direction   = glyph->direction    ? glyph->direction    + batch_start : nullptr;
                batch.glyph_half_length = glyph->half_length  ? glyph->half_length  + batch_start : nullptr;
                batch.glyph_sigma_x     = glyph->sigma_x      ? glyph->sigma_x      + batch_start : nullptr;
                batch.glyph_sigma_y     = glyph->sigma_y      ? glyph->sigma_y      + batch_start : nullptr;
                batch.glyph_rotation    = glyph->rotation     ? glyph->rotation     + batch_start : nullptr;
            }

            out_batches.push_back(batch);

            // Move to next batch
            if (i < num_points) {
                batch_start = i;
                // Skip invalid points
                while (batch_start < num_points && !assignment.valid_mask[batch_start]) {
                    ++batch_start;
                }
                // Update i to continue from the new batch_start position
                // This prevents underflow when i - batch_start is computed on next iteration
                i = batch_start;
                if (batch_start < num_points) {
                    current_tile = assignment.tile_indices[batch_start];
                }
            }
        }
    }

    return Status::success();
}

#ifdef PCR_HAS_CUDA

// ---------------------------------------------------------------------------
// GPU implementation
// ---------------------------------------------------------------------------
// These functions are only available when PCR_ENABLE_CUDA=ON.
// They handle Device and HostPinned memory locations.
// ---------------------------------------------------------------------------

/// GPU implementation of assign(): convert points to tile/cell indices
/// Calls CUDA kernel to compute tile/cell indices on device.
/// Allocates device memory for output arrays.
static Status assign_gpu_dispatch(const PointCloud& cloud, TileAssignment& out,
                                   const GridConfig& config, MemoryPool* pool, void* stream) {
    // Validate
    if (cloud.location() != MemoryLocation::Device &&
        cloud.location() != MemoryLocation::HostPinned) {
        return Status::error(StatusCode::InvalidArgument,
            "GPU tile router requires Device or HostPinned memory location");
    }

    size_t num_points = cloud.count();
    if (num_points == 0) {
        out.cell_indices = nullptr;
        out.tile_indices = nullptr;
        out.valid_mask = nullptr;
        out.num_points = 0;
        out.location = MemoryLocation::Device;
        return Status::success();
    }

    auto alloc = [&](size_t bytes) -> void* {
        if (pool) return pool->allocate(bytes);
        void* p = nullptr;
        cudaMalloc(&p, bytes);
        return p;
    };

    out.cell_indices = static_cast<uint32_t*>(alloc(num_points * sizeof(uint32_t)));
    out.tile_indices = static_cast<uint32_t*>(alloc(num_points * sizeof(uint32_t)));
    out.valid_mask   = static_cast<uint8_t*>(alloc(num_points * sizeof(uint8_t)));
    out.num_points   = num_points;

    if (!out.cell_indices || !out.tile_indices || !out.valid_mask) {
        return Status::error(StatusCode::OutOfMemory,
            "tile_router assign: allocation failed");
    }

    Status s = tile_router_assign_gpu(
        cloud.x(), cloud.y(), num_points,
        config,
        out.cell_indices, out.tile_indices, out.valid_mask,
        stream);

    if (s.ok()) {
        out.location = MemoryLocation::Device;
    }
    return s;
}

// ---------------------------------------------------------------------------
// Public API: Dispatchers (GPU build)
// ---------------------------------------------------------------------------
// These dispatchers route to CPU or GPU implementations based on memory location.
// This allows the same API to work with both Host and Device memory.
// ---------------------------------------------------------------------------

/// Public API dispatcher: routes to CPU or GPU based on cloud memory location
Status TileRouter::assign(const PointCloud& cloud, TileAssignment& out, void* stream) {
    if (cloud.location() == MemoryLocation::Host) {
        return assign_cpu(cloud, out, impl_->config, stream);
    } else {
        return assign_gpu_dispatch(cloud, out, impl_->config, impl_->pool, stream);
    }
}

/// GPU implementation of sort(): sort points by tile and cell indices
/// Uses CUB radix sort on device. Significantly faster than CPU for large datasets.
static Status sort_gpu_dispatch(TileAssignment& assignment,
                                 float* values,
                                 float* weights,
                                 float* timestamps,
                                 MemoryPool* pool,
                                 void* stream,
                                 GlyphSortArrays* glyph) {
    if (assignment.num_points == 0) return Status::success();

    return tile_router_sort_gpu(
        assignment.cell_indices, assignment.tile_indices,
        assignment.valid_mask,
        values, weights, timestamps,
        assignment.num_points,
        pool, stream, glyph);
}

/// Public API dispatcher: routes to CPU or GPU based on assignment memory location
Status TileRouter::sort(TileAssignment& assignment,
                        float* values,
                        float* weights,
                        float* timestamps,
                        GlyphSortArrays* glyph,
                        void* stream) {
    if (assignment.location == MemoryLocation::Host) {
        return sort_cpu(assignment, values, weights, timestamps,
                        glyph, impl_->config, impl_->sort_indices, stream);
    } else {
        return sort_gpu_dispatch(assignment, values, weights, timestamps,
                                 impl_->pool, stream, glyph);
    }
}

/// GPU implementation of extract_batches(): split sorted points into per-tile batches
/// Converts global-to-local cell indices on GPU, then copies tile boundaries to host.
/// Batches reference device memory pointers (no allocation needed for local_cell_indices).
static Status extract_batches_gpu_dispatch(const TileAssignment& assignment,
                                            float* sorted_values,
                                            float* sorted_weights,
                                            float* sorted_timestamps,
                                            std::vector<TileBatch>& out_batches,
                                            const GridConfig& config,
                                            GlyphSortArrays* glyph = nullptr) {
    out_batches.clear();
    size_t num_points = assignment.num_points;
    if (num_points == 0) return Status::success();

    // Convert global cell indices to local on GPU
    // (modifies cell_indices in-place)
    Status s = tile_router_global_to_local_gpu(
        assignment.cell_indices, assignment.tile_indices,
        num_points,
        config.width, config.height, config.tile_width, config.tile_height,
        nullptr);
    if (!s.ok()) return s;

    // Copy tile_indices and valid_mask to host to find batch boundaries
    std::vector<uint32_t> h_tiles(num_points);
    std::vector<uint8_t>  h_valid(num_points);
    cudaMemcpy(h_tiles.data(), assignment.tile_indices, num_points * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_valid.data(), assignment.valid_mask, num_points * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Find valid start
    size_t batch_start = 0;
    while (batch_start < num_points && !h_valid[batch_start]) {
        ++batch_start;
    }
    if (batch_start >= num_points) return Status::success();

    uint32_t current_tile = h_tiles[batch_start];
    int tiles_x = (config.width + config.tile_width - 1) / config.tile_width;

    for (size_t i = batch_start + 1; i <= num_points; ++i) {
        bool new_batch = (i == num_points) ||
                         (!h_valid[i]) ||
                         (h_tiles[i] != current_tile);

        if (new_batch) {
            size_t batch_size = i - batch_start;
            int tile_row = current_tile / tiles_x;
            int tile_col = current_tile % tiles_x;

            TileBatch batch;
            batch.tile = {tile_row, tile_col};
            batch.local_cell_indices = assignment.cell_indices + batch_start;
            batch.values = sorted_values ? (sorted_values + batch_start) : nullptr;
            batch.weights = sorted_weights ? (sorted_weights + batch_start) : nullptr;
            batch.timestamps = sorted_timestamps ? (sorted_timestamps + batch_start) : nullptr;
            batch.num_points = batch_size;
            batch.location = MemoryLocation::Device;

            // Propagate glyph array pointers (device memory offsets)
            if (glyph) {
                batch.coord_x           = glyph->coord_x     ? glyph->coord_x     + batch_start : nullptr;
                batch.coord_y           = glyph->coord_y     ? glyph->coord_y     + batch_start : nullptr;
                batch.glyph_direction   = glyph->direction   ? glyph->direction   + batch_start : nullptr;
                batch.glyph_half_length = glyph->half_length ? glyph->half_length + batch_start : nullptr;
                batch.glyph_sigma_x     = glyph->sigma_x     ? glyph->sigma_x     + batch_start : nullptr;
                batch.glyph_sigma_y     = glyph->sigma_y     ? glyph->sigma_y     + batch_start : nullptr;
                batch.glyph_rotation    = glyph->rotation    ? glyph->rotation    + batch_start : nullptr;
            }

            out_batches.push_back(batch);

            if (i < num_points) {
                batch_start = i;
                while (batch_start < num_points && !h_valid[batch_start]) {
                    ++batch_start;
                }
                if (batch_start >= num_points) break;
                current_tile = h_tiles[batch_start];
                i = batch_start;  // resume outer loop from batch_start+1
            }
        }
    }

    return Status::success();
}

/// Public API dispatcher: routes to CPU or GPU based on assignment memory location
Status TileRouter::extract_batches(const TileAssignment& assignment,
                                   float* sorted_values,
                                   float* sorted_weights,
                                   float* sorted_timestamps,
                                   std::vector<TileBatch>& out_batches,
                                   GlyphSortArrays* glyph) {
    if (assignment.location == MemoryLocation::Host) {
        return extract_batches_cpu(assignment, sorted_values, sorted_weights,
                                   sorted_timestamps, out_batches, impl_->config, glyph);
    } else {
        return extract_batches_gpu_dispatch(assignment, sorted_values, sorted_weights,
                                            sorted_timestamps, out_batches, impl_->config, glyph);
    }
}

#else

// ---------------------------------------------------------------------------
// CPU-only build (PCR_ENABLE_CUDA=OFF)
// ---------------------------------------------------------------------------
// When CUDA is disabled, public API always routes to CPU implementations.
// GPU memory locations (Device, HostPinned) will fail with InvalidArgument.
// ---------------------------------------------------------------------------

/// Public API: CPU-only - always use CPU implementation
Status TileRouter::assign(const PointCloud& cloud, TileAssignment& out, void* stream) {
    return assign_cpu(cloud, out, impl_->config, stream);
}

/// Public API: CPU-only - always use CPU implementation
Status TileRouter::sort(TileAssignment& assignment,
                        float* values,
                        float* weights,
                        float* timestamps,
                        GlyphSortArrays* glyph,
                        void* stream) {
    return sort_cpu(assignment, values, weights, timestamps,
                    glyph, impl_->config, impl_->sort_indices, stream);
}

/// Public API: CPU-only - always use CPU implementation
Status TileRouter::extract_batches(const TileAssignment& assignment,
                                   float* sorted_values,
                                   float* sorted_weights,
                                   float* sorted_timestamps,
                                   std::vector<TileBatch>& out_batches,
                                   GlyphSortArrays* glyph) {
    return extract_batches_cpu(assignment, sorted_values, sorted_weights,
                               sorted_timestamps, out_batches, impl_->config, glyph);
}

#endif // PCR_HAS_CUDA

} // namespace pcr
