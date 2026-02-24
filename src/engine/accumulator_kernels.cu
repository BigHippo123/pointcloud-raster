#include "pcr/engine/accumulator_kernels.h"

#ifdef PCR_HAS_CUDA
#include "pcr/ops/builtin_ops.h"
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>

namespace pcr {

// ---------------------------------------------------------------------------
// Strategy: Direct scatter approach (works for all ops)
//
// For each point, atomically update the cell state. Since points are sorted
// by cell index, we can use a segmented approach: process runs of same-cell
// points sequentially within a warp, then scatter the result.
//
// Simpler initial approach: one thread per point, use atomics for Sum/Count,
// and a lock-free approach for others.
//
// For correctness and generality, we use a simple per-point approach with
// atomic operations where possible, and a serial-within-cell approach for
// multi-state ops.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Kernel: Accumulate using direct atomic scatter (for 1-float state ops)
// ---------------------------------------------------------------------------

__global__ void kernel_accumulate_sum(
    const uint32_t* cell_indices, const float* values,
    float* state, size_t num_points, int64_t tile_cells)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    uint32_t cell = cell_indices[i];
    if (cell < static_cast<uint32_t>(tile_cells)) {
        atomicAdd(&state[cell], values[i]);
    }
}

__global__ void kernel_accumulate_count(
    const uint32_t* cell_indices,
    float* state, size_t num_points, int64_t tile_cells)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    uint32_t cell = cell_indices[i];
    if (cell < static_cast<uint32_t>(tile_cells)) {
        atomicAdd(&state[cell], 1.0f);
    }
}

__global__ void kernel_accumulate_max(
    const uint32_t* cell_indices, const float* values,
    float* state, size_t num_points, int64_t tile_cells)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    uint32_t cell = cell_indices[i];
    if (cell >= static_cast<uint32_t>(tile_cells)) return;

    // Atomic max for float using atomicCAS
    float val = values[i];
    unsigned int* addr = reinterpret_cast<unsigned int*>(&state[cell]);
    unsigned int old = *addr, assumed;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val >= val) break;
        old = atomicCAS(addr, assumed, __float_as_uint(val));
    } while (assumed != old);
}

__global__ void kernel_accumulate_min(
    const uint32_t* cell_indices, const float* values,
    float* state, size_t num_points, int64_t tile_cells)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    uint32_t cell = cell_indices[i];
    if (cell >= static_cast<uint32_t>(tile_cells)) return;

    float val = values[i];
    unsigned int* addr = reinterpret_cast<unsigned int*>(&state[cell]);
    unsigned int old = *addr, assumed;
    do {
        assumed = old;
        float old_val = __uint_as_float(assumed);
        if (old_val <= val) break;
        old = atomicCAS(addr, assumed, __float_as_uint(val));
    } while (assumed != old);
}

// ---------------------------------------------------------------------------
// Kernel: Accumulate for 2-float state ops (Average, WeightedAverage)
// Uses atomicAdd on both state fields.
// ---------------------------------------------------------------------------

__global__ void kernel_accumulate_average(
    const uint32_t* cell_indices, const float* values,
    float* state, size_t num_points, int64_t tile_cells)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    uint32_t cell = cell_indices[i];
    if (cell < static_cast<uint32_t>(tile_cells)) {
        atomicAdd(&state[cell], values[i]);                // sum band
        atomicAdd(&state[tile_cells + cell], 1.0f);        // count band
    }
}

__global__ void kernel_accumulate_weighted_average(
    const uint32_t* cell_indices, const float* values,
    const float* weights,
    float* state, size_t num_points, int64_t tile_cells)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    uint32_t cell = cell_indices[i];
    if (cell < static_cast<uint32_t>(tile_cells)) {
        float w = weights[i];
        atomicAdd(&state[cell], values[i] * w);             // weighted_sum band
        atomicAdd(&state[tile_cells + cell], w);             // weight_sum band
    }
}

// ---------------------------------------------------------------------------
// Kernel: MostRecent — need compare-and-swap on timestamp, conditional value
// ---------------------------------------------------------------------------

__global__ void kernel_accumulate_most_recent(
    const uint32_t* cell_indices, const float* values,
    const float* timestamps,
    float* state, size_t num_points, int64_t tile_cells)
{
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    uint32_t cell = cell_indices[i];
    if (cell >= static_cast<uint32_t>(tile_cells)) return;

    float val = values[i];
    float ts  = timestamps[i];

    // Atomic CAS loop on timestamp (band 1)
    unsigned int* ts_addr = reinterpret_cast<unsigned int*>(&state[tile_cells + cell]);
    unsigned int old_ts = *ts_addr, assumed;
    do {
        assumed = old_ts;
        float old_ts_val = __uint_as_float(assumed);
        if (old_ts_val >= ts) break;  // existing timestamp is newer
        old_ts = atomicCAS(ts_addr, assumed, __float_as_uint(ts));
        if (old_ts == assumed) {
            // We won the CAS, update value
            state[cell] = val;
            break;
        }
    } while (true);
}

// ---------------------------------------------------------------------------
// Host dispatch function
// ---------------------------------------------------------------------------

static constexpr int BLOCK_SIZE = 256;

static int grid_dim(size_t n) {
    return static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

Status accumulate_gpu(
    ReductionType type,
    const uint32_t* d_cell_indices,
    const float*    d_values,
    const float*    d_weights,
    const float*    d_timestamps,
    size_t          num_points,
    float*          d_state,
    int64_t         tile_cells,
    MemoryPool*     pool,
    void*           stream)
{
    if (num_points == 0) return Status::success();

    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int blocks = grid_dim(num_points);

    switch (type) {
        case ReductionType::Sum:
            kernel_accumulate_sum<<<blocks, BLOCK_SIZE, 0, s>>>(
                d_cell_indices, d_values, d_state, num_points, tile_cells);
            break;

        case ReductionType::Max:
            kernel_accumulate_max<<<blocks, BLOCK_SIZE, 0, s>>>(
                d_cell_indices, d_values, d_state, num_points, tile_cells);
            break;

        case ReductionType::Min:
            kernel_accumulate_min<<<blocks, BLOCK_SIZE, 0, s>>>(
                d_cell_indices, d_values, d_state, num_points, tile_cells);
            break;

        case ReductionType::Count:
            kernel_accumulate_count<<<blocks, BLOCK_SIZE, 0, s>>>(
                d_cell_indices, d_state, num_points, tile_cells);
            break;

        case ReductionType::Average:
            kernel_accumulate_average<<<blocks, BLOCK_SIZE, 0, s>>>(
                d_cell_indices, d_values, d_state, num_points, tile_cells);
            break;

        case ReductionType::WeightedAverage:
            if (!d_weights) {
                return Status::error(StatusCode::InvalidArgument,
                    "accumulate_gpu: WeightedAverage requires weights");
            }
            kernel_accumulate_weighted_average<<<blocks, BLOCK_SIZE, 0, s>>>(
                d_cell_indices, d_values, d_weights, d_state, num_points, tile_cells);
            break;

        case ReductionType::MostRecent:
            if (!d_timestamps) {
                return Status::error(StatusCode::InvalidArgument,
                    "accumulate_gpu: MostRecent requires timestamps");
            }
            kernel_accumulate_most_recent<<<blocks, BLOCK_SIZE, 0, s>>>(
                d_cell_indices, d_values, d_timestamps, d_state, num_points, tile_cells);
            break;

        default:
            return Status::error(StatusCode::InvalidArgument,
                "accumulate_gpu: unsupported reduction type");
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return Status::error(StatusCode::CudaError,
            std::string("accumulate_gpu: ") + cudaGetErrorString(err));
    }
    return Status::success();
}

} // namespace pcr

#endif // PCR_HAS_CUDA
