#include "pcr/engine/tile_router_kernels.h"

#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

namespace pcr {

// ---------------------------------------------------------------------------
// Device-side GridConfig (POD subset passed by value to kernels)
// ---------------------------------------------------------------------------
struct DeviceGridConfig {
    double origin_x;
    double origin_y;
    double cell_size_x;
    double cell_size_y;
    int    width;
    int    height;
    int    tile_width;
    int    tile_height;
    int    tiles_x;
};

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

static constexpr int BLOCK_SIZE = 256;

static int grid_size(size_t n) {
    return static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

__global__ void kernel_assign(
    const double* x, const double* y, size_t num_points,
    DeviceGridConfig cfg,
    uint32_t* cell_indices, uint32_t* tile_indices, uint8_t* valid_mask)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    double px = x[i];
    double py = y[i];

    int cx = static_cast<int>((px - cfg.origin_x) / cfg.cell_size_x);
    int cy = static_cast<int>((py - cfg.origin_y) / cfg.cell_size_y);

    if (cx < 0 || cy < 0 || cx >= cfg.width || cy >= cfg.height) {
        valid_mask[i] = 0;
        cell_indices[i] = 0;
        tile_indices[i] = 0;
        return;
    }

    valid_mask[i] = 1;
    cell_indices[i] = static_cast<uint32_t>(cy * cfg.width + cx);

    int tile_col = cx / cfg.tile_width;
    int tile_row = cy / cfg.tile_height;
    tile_indices[i] = static_cast<uint32_t>(tile_row * cfg.tiles_x + tile_col);
}

__global__ void kernel_build_sort_keys(
    const uint32_t* tile_indices, const uint32_t* cell_indices,
    const uint8_t* valid_mask,
    uint64_t* keys, uint32_t* iota, size_t n)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t tile = valid_mask[i] ? static_cast<uint64_t>(tile_indices[i]) : 0xFFFFFFFFULL;
    keys[i] = (tile << 32) | static_cast<uint64_t>(cell_indices[i]);
    iota[i] = static_cast<uint32_t>(i);
}

__global__ void kernel_apply_perm_u32(
    const uint32_t* src, uint32_t* dst, const uint32_t* perm, size_t n)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[perm[i]];
}

__global__ void kernel_apply_perm_u8(
    const uint8_t* src, uint8_t* dst, const uint32_t* perm, size_t n)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[perm[i]];
}

__global__ void kernel_apply_perm_f32(
    const float* src, float* dst, const uint32_t* perm, size_t n)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[perm[i]];
}

__global__ void kernel_global_to_local(
    uint32_t* cell_indices, const uint32_t* tile_indices,
    size_t num_points, int grid_width, int grid_height, int tile_width, int tile_height, int tiles_x)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    uint32_t global_cell = cell_indices[i];
    uint32_t tile_idx = tile_indices[i];

    int cx = global_cell % grid_width;
    int cy = global_cell / grid_width;

    int tile_col = tile_idx % tiles_x;
    int tile_row = tile_idx / tiles_x;

    int tile_x0 = tile_col * tile_width;
    int tile_y0 = tile_row * tile_height;

    // Calculate actual tile dimensions (may be smaller for edge tiles)
    int actual_tile_width = min(tile_width, grid_width - tile_x0);
    int actual_tile_height = min(tile_height, grid_height - tile_y0);

    int local_cx = cx - tile_x0;
    int local_cy = cy - tile_y0;

    // Use actual tile width for local cell index calculation
    cell_indices[i] = static_cast<uint32_t>(local_cy * actual_tile_width + local_cx);
}

// ---------------------------------------------------------------------------
// Host functions
// ---------------------------------------------------------------------------

Status tile_router_assign_gpu(
    const double* d_x, const double* d_y, size_t num_points,
    const GridConfig& config,
    uint32_t* d_cell_indices, uint32_t* d_tile_indices, uint8_t* d_valid_mask,
    void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    DeviceGridConfig dcfg;
    dcfg.origin_x    = config.bounds.min_x;
    dcfg.origin_y    = config.bounds.max_y;  // north-up origin at top-left
    dcfg.cell_size_x = config.cell_size_x;
    dcfg.cell_size_y = config.cell_size_y;
    dcfg.width       = config.width;
    dcfg.height      = config.height;
    dcfg.tile_width  = config.tile_width;
    dcfg.tile_height = config.tile_height;
    dcfg.tiles_x     = (config.width + config.tile_width - 1) / config.tile_width;

    kernel_assign<<<grid_size(num_points), BLOCK_SIZE, 0, s>>>(
        d_x, d_y, num_points, dcfg,
        d_cell_indices, d_tile_indices, d_valid_mask);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            std::string("tile_router_assign: ") + cudaGetErrorString(err));
    }
    return Status::success();
}

Status tile_router_sort_gpu(
    uint32_t* d_cell_indices, uint32_t* d_tile_indices,
    uint8_t* d_valid_mask,
    float* d_values, float* d_weights, float* d_timestamps,
    size_t num_points, MemoryPool* pool, void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int blocks = grid_size(num_points);

    auto alloc = [&](size_t bytes) -> void* {
        if (pool) return pool->allocate(bytes);
        void* p = nullptr;
        cudaMalloc(&p, bytes);
        return p;
    };

    // Allocate temp buffers
    uint64_t* d_keys_in  = static_cast<uint64_t*>(alloc(num_points * sizeof(uint64_t)));
    uint64_t* d_keys_out = static_cast<uint64_t*>(alloc(num_points * sizeof(uint64_t)));
    uint32_t* d_perm_in  = static_cast<uint32_t*>(alloc(num_points * sizeof(uint32_t)));
    uint32_t* d_perm_out = static_cast<uint32_t*>(alloc(num_points * sizeof(uint32_t)));

    if (!d_keys_in || !d_keys_out || !d_perm_in || !d_perm_out) {
        return Status::error(StatusCode::OutOfMemory, "tile_router_sort: allocation failed");
    }

    // Build composite keys and iota permutation
    kernel_build_sort_keys<<<blocks, BLOCK_SIZE, 0, s>>>(
        d_tile_indices, d_cell_indices, d_valid_mask,
        d_keys_in, d_perm_in, num_points);

    // CUB radix sort (key-value sort on composite key)
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_bytes,
        d_keys_in, d_keys_out, d_perm_in, d_perm_out,
        static_cast<int>(num_points), 0, 64, s);

    void* d_temp = alloc(temp_bytes);
    if (!d_temp) {
        return Status::error(StatusCode::OutOfMemory, "tile_router_sort: CUB temp alloc failed");
    }

    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_keys_in, d_keys_out, d_perm_in, d_perm_out,
        static_cast<int>(num_points), 0, 64, s);

    // Apply permutation to all arrays (we need temp copies of the originals)
    // cell_indices
    uint32_t* d_cell_tmp = static_cast<uint32_t*>(alloc(num_points * sizeof(uint32_t)));
    uint32_t* d_tile_tmp = static_cast<uint32_t*>(alloc(num_points * sizeof(uint32_t)));
    uint8_t*  d_valid_tmp = static_cast<uint8_t*>(alloc(num_points * sizeof(uint8_t)));

    if (!d_cell_tmp || !d_tile_tmp || !d_valid_tmp) {
        return Status::error(StatusCode::OutOfMemory, "tile_router_sort: perm alloc failed");
    }

    // Copy originals to temps
    cudaMemcpyAsync(d_cell_tmp, d_cell_indices, num_points * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(d_tile_tmp, d_tile_indices, num_points * sizeof(uint32_t), cudaMemcpyDeviceToDevice, s);
    cudaMemcpyAsync(d_valid_tmp, d_valid_mask, num_points * sizeof(uint8_t), cudaMemcpyDeviceToDevice, s);

    // Apply permutation
    kernel_apply_perm_u32<<<blocks, BLOCK_SIZE, 0, s>>>(d_cell_tmp, d_cell_indices, d_perm_out, num_points);
    kernel_apply_perm_u32<<<blocks, BLOCK_SIZE, 0, s>>>(d_tile_tmp, d_tile_indices, d_perm_out, num_points);
    kernel_apply_perm_u8<<<blocks, BLOCK_SIZE, 0, s>>>(d_valid_tmp, d_valid_mask, d_perm_out, num_points);

    // Co-sort value arrays
    if (d_values) {
        float* d_val_tmp = static_cast<float*>(alloc(num_points * sizeof(float)));
        if (!d_val_tmp) return Status::error(StatusCode::OutOfMemory, "tile_router_sort: values alloc failed");
        cudaMemcpyAsync(d_val_tmp, d_values, num_points * sizeof(float), cudaMemcpyDeviceToDevice, s);
        kernel_apply_perm_f32<<<blocks, BLOCK_SIZE, 0, s>>>(d_val_tmp, d_values, d_perm_out, num_points);
    }
    if (d_weights) {
        float* d_wt_tmp = static_cast<float*>(alloc(num_points * sizeof(float)));
        if (!d_wt_tmp) return Status::error(StatusCode::OutOfMemory, "tile_router_sort: weights alloc failed");
        cudaMemcpyAsync(d_wt_tmp, d_weights, num_points * sizeof(float), cudaMemcpyDeviceToDevice, s);
        kernel_apply_perm_f32<<<blocks, BLOCK_SIZE, 0, s>>>(d_wt_tmp, d_weights, d_perm_out, num_points);
    }
    if (d_timestamps) {
        float* d_ts_tmp = static_cast<float*>(alloc(num_points * sizeof(float)));
        if (!d_ts_tmp) return Status::error(StatusCode::OutOfMemory, "tile_router_sort: timestamps alloc failed");
        cudaMemcpyAsync(d_ts_tmp, d_timestamps, num_points * sizeof(float), cudaMemcpyDeviceToDevice, s);
        kernel_apply_perm_f32<<<blocks, BLOCK_SIZE, 0, s>>>(d_ts_tmp, d_timestamps, d_perm_out, num_points);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            std::string("tile_router_sort: ") + cudaGetErrorString(err));
    }
    return Status::success();
}

Status tile_router_global_to_local_gpu(
    uint32_t* d_cell_indices, const uint32_t* d_tile_indices,
    size_t num_points,
    int grid_width, int grid_height, int tile_width, int tile_height,
    void* stream)
{
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int tiles_x = (grid_width + tile_width - 1) / tile_width;

    kernel_global_to_local<<<grid_size(num_points), BLOCK_SIZE, 0, s>>>(
        d_cell_indices, d_tile_indices, num_points,
        grid_width, grid_height, tile_width, tile_height, tiles_x);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            std::string("tile_router_global_to_local: ") + cudaGetErrorString(err));
    }
    return Status::success();
}

} // namespace pcr

#endif // PCR_HAS_CUDA
