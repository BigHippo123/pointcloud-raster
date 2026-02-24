#include "pcr/engine/pipeline.h"
#include "pcr/engine/tile_manager.h"
#include "pcr/engine/tile_router.h"
#include "pcr/engine/accumulator.h"
#include "pcr/engine/filter.h"
#include "pcr/engine/memory_pool.h"
#include "pcr/core/grid.h"
#include "pcr/core/point_cloud.h"
#include "pcr/ops/reduction_registry.h"
#include "pcr/io/grid_io.h"
#include <chrono>
#include <cstring>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#endif
#ifdef PCR_HAS_OPENMP
#include <omp.h>
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// Pipeline::Impl
// ---------------------------------------------------------------------------
struct Pipeline::Impl {
    PipelineConfig config;
    ProgressCallback progress_callback;

    // Engine components
    std::vector<std::unique_ptr<TileManager>> tile_managers;  // One per reduction
    std::unique_ptr<TileRouter> router;
    std::unique_ptr<TileRouter> cpu_router;   // CPU-only router for Hybrid mode
    std::unique_ptr<Accumulator> accumulator;
    std::unique_ptr<MemoryPool> memory_pool;  // GPU memory pool (if using GPU)

    // GPU resources
#ifdef PCR_HAS_CUDA
    cudaStream_t cuda_stream = nullptr;
#endif

    // Results
    std::unique_ptr<Grid> result_grid;

    // Stats
    size_t collections_processed = 0;
    size_t points_processed = 0;
    std::chrono::steady_clock::time_point start_time;

    Impl(const PipelineConfig& cfg) : config(cfg) {
        start_time = std::chrono::steady_clock::now();
    }

    // Check if we should use GPU based on config and cloud location
    bool should_use_gpu(const PointCloud& cloud) const {
#ifdef PCR_HAS_CUDA
        switch (config.exec_mode) {
            case ExecutionMode::CPU:
                return false;
            case ExecutionMode::GPU:
                return true;
            case ExecutionMode::Auto:
                // Auto: use GPU if the pool was initialized (GPU is available).
                // Tile state location is fixed at init time, so we must commit to
                // one memory space for the entire pipeline run.
                return memory_pool != nullptr;
            case ExecutionMode::Hybrid:
                // Hybrid: tile state is on GPU (same as GPU mode).
                // Routing is CPU-side; accumulation uses GPU.
                return memory_pool != nullptr;
        }
        return false;
#else
        (void)cloud;
        return false;
#endif
    }

    ~Impl() {
#ifdef PCR_HAS_CUDA
        if (cuda_stream) {
            cudaStreamDestroy(cuda_stream);
        }
#endif
    }

    Status initialize() {
        // Configure CPU threading
#ifdef PCR_HAS_OPENMP
        if (config.cpu_threads > 0) {
            omp_set_num_threads(static_cast<int>(config.cpu_threads));
        }
#endif

#ifndef PCR_HAS_CUDA
        // Without CUDA, Hybrid falls back to CPU mode silently
        if (config.exec_mode == ExecutionMode::Hybrid) {
            fprintf(stderr, "Info: Hybrid mode requested but CUDA not available - using CPU mode\n");
            const_cast<PipelineConfig&>(config).exec_mode = ExecutionMode::CPU;
        }
#endif

        // Create GPU resources if needed (must be done before tile managers)
#ifdef PCR_HAS_CUDA
        if (config.exec_mode == ExecutionMode::GPU || config.exec_mode == ExecutionMode::Auto ||
            config.exec_mode == ExecutionMode::Hybrid) {
            // Check if GPU is available
            if (!cuda_device_available()) {
                std::string msg = "No CUDA-capable GPU detected";

                if (config.exec_mode == ExecutionMode::GPU) {
                    if (config.gpu_require_strict) {
                        return Status::error(StatusCode::CudaError,
                            msg + " - GPU mode requested but no GPU available");
                    } else if (config.gpu_fallback_to_cpu) {
                        // Fall back to CPU
                        fprintf(stderr, "Warning: %s - falling back to CPU mode\n", msg.c_str());
                        const_cast<PipelineConfig&>(config).exec_mode = ExecutionMode::CPU;
                    } else {
                        return Status::error(StatusCode::CudaError,
                            msg + " - GPU required but not available");
                    }
                } else {
                    // Auto mode: silently use CPU if no GPU
                    fprintf(stderr, "Info: %s - using CPU mode\n", msg.c_str());
                }
            } else {
                // Set CUDA device
                if (config.cuda_device_id >= 0) {
                    cudaError_t err = cudaSetDevice(config.cuda_device_id);
                    if (err != cudaSuccess) {
                        std::string err_msg = std::string("Failed to set CUDA device ") +
                            std::to_string(config.cuda_device_id) + ": " + cudaGetErrorString(err);

                        if (config.gpu_require_strict) {
                            return Status::error(StatusCode::CudaError, err_msg);
                        } else if (config.gpu_fallback_to_cpu) {
                            fprintf(stderr, "Warning: %s - falling back to CPU mode\n", err_msg.c_str());
                            const_cast<PipelineConfig&>(config).exec_mode = ExecutionMode::CPU;
                        } else {
                            return Status::error(StatusCode::CudaError, err_msg);
                        }
                    } else {
                        // Log successful GPU initialization
                        std::string device_name = cuda_device_name(config.cuda_device_id);
                        size_t free_mem = 0, total_mem = 0;
                        if (cuda_get_memory_info(&free_mem, &total_mem, config.cuda_device_id)) {
                            fprintf(stderr, "Info: Using GPU %d: %s (%.1f GB free / %.1f GB total)\n",
                                config.cuda_device_id, device_name.c_str(),
                                free_mem / (1024.0 * 1024.0 * 1024.0),
                                total_mem / (1024.0 * 1024.0 * 1024.0));
                        } else {
                            fprintf(stderr, "Info: Using GPU %d: %s\n",
                                config.cuda_device_id, device_name.c_str());
                        }
                    }
                }

                // Only create GPU resources if we successfully set the device
                if (config.exec_mode == ExecutionMode::GPU || config.exec_mode == ExecutionMode::Auto ||
                    config.exec_mode == ExecutionMode::Hybrid) {
                    // Determine effective pool size:
                    //   1. If gpu_memory_budget > 0, use it directly.
                    //   2. Otherwise auto-detect: use 80% of free GPU memory,
                    //      but at least gpu_pool_size_bytes (512MB default).
                    size_t pool_bytes = config.gpu_pool_size_bytes;
                    if (config.gpu_memory_budget > 0) {
                        pool_bytes = config.gpu_memory_budget;
                    } else {
                        size_t free_mem = 0, total_mem = 0;
                        if (cuda_get_memory_info(&free_mem, &total_mem, config.cuda_device_id)) {
                            size_t auto_size = static_cast<size_t>(free_mem * 0.8);
                            if (auto_size > pool_bytes) {
                                pool_bytes = auto_size;
                            }
                        }
                    }
                    // Create memory pool
                    memory_pool = MemoryPool::create(pool_bytes);
                    if (!memory_pool) {
                        std::string err_msg = "Failed to create GPU memory pool";
                        if (config.gpu_require_strict) {
                            return Status::error(StatusCode::OutOfMemory, err_msg);
                        } else if (config.gpu_fallback_to_cpu) {
                            fprintf(stderr, "Warning: %s - falling back to CPU mode\n", err_msg.c_str());
                            const_cast<PipelineConfig&>(config).exec_mode = ExecutionMode::CPU;
                        } else {
                            return Status::error(StatusCode::OutOfMemory, err_msg);
                        }
                    }

                    // Create CUDA stream
                    if (memory_pool && config.use_cuda_streams) {
                        cudaError_t err = cudaStreamCreate(&cuda_stream);
                        if (err != cudaSuccess) {
                            std::string err_msg = std::string("Failed to create CUDA stream: ") +
                                cudaGetErrorString(err);
                            if (config.gpu_require_strict) {
                                return Status::error(StatusCode::CudaError, err_msg);
                            } else if (config.gpu_fallback_to_cpu) {
                                fprintf(stderr, "Warning: %s - falling back to CPU mode\n", err_msg.c_str());
                                const_cast<PipelineConfig&>(config).exec_mode = ExecutionMode::CPU;
                                memory_pool.reset();  // Clean up memory pool
                            } else {
                                return Status::error(StatusCode::CudaError, err_msg);
                            }
                        }
                    }
                }
            }
        }
#endif

        // Determine memory location for tile state based on execution mode
        MemoryLocation tile_memory_location = MemoryLocation::Host;
#ifdef PCR_HAS_CUDA
        if (memory_pool) {  // memory_pool exists iff GPU is active (GPU or Auto with GPU)
            tile_memory_location = MemoryLocation::Device;
        }
#endif

        // Create tile managers (one per reduction) - after GPU resources are ready
        for (const auto& reduction : config.reductions) {
            const ReductionInfo* info = get_reduction(reduction.type);
            if (!info) {
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: unknown reduction type");
            }

            TileManagerConfig tm_config;
            tm_config.state_dir = config.state_dir.empty() ? "/tmp/pcr_tiles" : config.state_dir;
            tm_config.cache_size_bytes = config.host_cache_budget > 0
                ? config.host_cache_budget
                : (size_t)1024 * 1024 * 1024;  // 1GB default
            tm_config.state_floats = info->state_floats;
            tm_config.grid_config = config.grid;
            tm_config.memory_location = tile_memory_location;
#ifdef PCR_HAS_CUDA
            tm_config.cuda_stream = cuda_stream;
#endif

            auto mgr = TileManager::create(tm_config);
            if (!mgr) {
                return Status::error(StatusCode::IoError,
                    "pipeline: failed to create tile manager");
            }

            tile_managers.push_back(std::move(mgr));
        }

        // Create tile router with optional memory pool
        router = TileRouter::create(config.grid, memory_pool.get());
        if (!router) {
            return Status::error(StatusCode::InvalidArgument,
                "pipeline: failed to create tile router");
        }

        // For Hybrid mode, also create a CPU-only router for the routing phase.
        // This routes on CPU threads while the GPU handles accumulation.
        if (config.exec_mode == ExecutionMode::Hybrid) {
            cpu_router = TileRouter::create(config.grid, nullptr);  // nullptr = CPU path
            if (!cpu_router) {
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: failed to create CPU router for Hybrid mode");
            }
        }

        // Create accumulator with optional memory pool
        accumulator = Accumulator::create(memory_pool.get());
        if (!accumulator) {
            return Status::error(StatusCode::OutOfMemory,
                "pipeline: failed to create accumulator");
        }

        return Status::success();
    }

    Status process_cloud(const PointCloud& cloud) {
        size_t num_points = cloud.count();
        if (num_points == 0) {
            return Status::success();  // Nothing to process
        }

        // Dispatch to Hybrid mode handler if configured
        if (config.exec_mode == ExecutionMode::Hybrid) {
            return process_cloud_hybrid(cloud);
        }

        // Transfer to device if using GPU and cloud is on Host
        std::unique_ptr<PointCloud> device_cloud;
        const PointCloud* processing_cloud = &cloud;

#ifdef PCR_HAS_CUDA
        if (should_use_gpu(cloud) && cloud.location() == MemoryLocation::Host) {
            // Async transfer to device
            device_cloud = cloud.to_device_async(cuda_stream);
            if (!device_cloud) {
                // Get more context about the failure
                size_t free_mem = 0, total_mem = 0;
                std::string mem_info = "";
                if (cuda_get_memory_info(&free_mem, &total_mem)) {
                    mem_info = " (GPU has " + std::to_string(free_mem / (1024*1024)) +
                              " MB free / " + std::to_string(total_mem / (1024*1024)) + " MB total)";
                }

                return Status::error(StatusCode::CudaError,
                    "pipeline: failed to transfer " + std::to_string(num_points) +
                    " points to GPU device" + mem_info +
                    " - try reducing batch size or using CPU mode");
            }
            processing_cloud = device_cloud.get();

            // Synchronize to ensure transfer completes before processing
            if (cuda_stream) {
                cudaError_t err = cudaStreamSynchronize(cuda_stream);
                if (err != cudaSuccess) {
                    return Status::error(StatusCode::CudaError,
                        std::string("CUDA stream sync failed: ") + cudaGetErrorString(err) +
                        " - GPU may be out of memory or in an error state");
                }
            }
        }
#endif

        // Note: The current implementation below uses CPU-centric data flow (std::vector).
        // Full GPU data flow (keeping data on device throughout) requires updating
        // the router and accumulator calls to use device memory throughout.
        // For now, this allows processing of clouds already on device, but may
        // transfer data back to host during intermediate steps.

        // Apply filter if specified
        uint32_t* filtered_indices = nullptr;
        size_t filtered_count = num_points;
        bool need_free_indices = false;

        if (!config.filter.empty()) {
            Status s = filter_points(*processing_cloud, config.filter, &filtered_indices, &filtered_count);
            if (!s.ok()) {
                return s;
            }
            need_free_indices = true;
        }

        // If all points filtered out, nothing to do
        if (filtered_count == 0) {
            if (need_free_indices) free(filtered_indices);
            return Status::success();
        }

        // Create filtered cloud view if needed
        // For now, we'll work with the full cloud and skip filtered points
        // A more efficient implementation would create a view

        // Process each reduction
        for (size_t r = 0; r < config.reductions.size(); ++r) {
            const auto& reduction = config.reductions[r];
            TileManager* mgr = tile_managers[r].get();

            // Get value channel
            const void* value_data = processing_cloud->channel_data(reduction.value_channel);
            if (!value_data) {
                if (need_free_indices) free(filtered_indices);
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: value channel not found: " + reduction.value_channel);
            }

            // For now, only support Float32 channels
            const ChannelDesc* desc = processing_cloud->channel(reduction.value_channel);
            if (!desc || desc->dtype != DataType::Float32) {
                if (need_free_indices) free(filtered_indices);
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: value channel must be Float32");
            }

            const float* values = static_cast<const float*>(value_data);

            // Copy values (or filtered subset).
            // When using GPU, `values` may be in device memory (if processing_cloud is on
            // Device). We must allocate a device buffer and use cudaMemcpy rather than
            // std::memcpy, which cannot access device pointers from the CPU.
            size_t copy_count = need_free_indices ? filtered_count : num_points;
            float* values_copy_raw = nullptr;
            bool values_copy_on_device = false;

#ifdef PCR_HAS_CUDA
            if (should_use_gpu(*processing_cloud)) {
                // Allocate device memory: GPU sort/accumulate kernels require device pointers.
                cudaError_t cerr = cudaMalloc(
                    reinterpret_cast<void**>(&values_copy_raw),
                    copy_count * sizeof(float));
                if (cerr != cudaSuccess) {
                    if (need_free_indices) free(filtered_indices);
                    return Status::error(StatusCode::OutOfMemory,
                        std::string("pipeline: failed to allocate device values buffer: ") +
                        cudaGetErrorString(cerr));
                }
                values_copy_on_device = true;

                if (need_free_indices) {
                    // filtered_indices may also be in device memory (filter_points_gpu).
                    // Copy values and indices to host, gather, then copy result to device.
                    std::vector<float> h_values(num_points);
                    std::vector<uint32_t> h_indices(filtered_count);
                    cudaMemcpy(h_values.data(), values,
                               num_points * sizeof(float), cudaMemcpyDefault);
                    cudaMemcpy(h_indices.data(), filtered_indices,
                               filtered_count * sizeof(uint32_t), cudaMemcpyDefault);
                    std::vector<float> h_filtered(filtered_count);
                    for (size_t i = 0; i < filtered_count; ++i) {
                        h_filtered[i] = h_values[h_indices[i]];
                    }
                    cudaMemcpy(values_copy_raw, h_filtered.data(),
                               filtered_count * sizeof(float), cudaMemcpyHostToDevice);
                } else {
                    // Direct copy from device (or HostPinned) to device buffer.
                    // cudaMemcpyDefault detects direction via UVA.
                    cudaMemcpy(values_copy_raw, values,
                               num_points * sizeof(float), cudaMemcpyDefault);
                }
            } else
#endif
            {
                // CPU path: use host memory and standard copies.
                values_copy_raw = static_cast<float*>(malloc(copy_count * sizeof(float)));
                if (!values_copy_raw) {
                    if (need_free_indices) free(filtered_indices);
                    return Status::error(StatusCode::OutOfMemory,
                        "pipeline: failed to allocate values buffer");
                }
                if (need_free_indices) {
                    for (size_t i = 0; i < filtered_count; ++i) {
                        values_copy_raw[i] = values[filtered_indices[i]];
                    }
                } else {
                    std::memcpy(values_copy_raw, values, num_points * sizeof(float));
                }
            }
            // RAII guard: frees values_copy_raw on all exit paths from this loop body
            struct ValuesGuard {
                float* ptr; bool on_device;
                ValuesGuard(float* p, bool dev) : ptr(p), on_device(dev) {}
                ~ValuesGuard() {
                    if (!ptr) return;
#ifdef PCR_HAS_CUDA
                    if (on_device) { cudaFree(ptr); return; }
#endif
                    free(ptr);
                }
                ValuesGuard(const ValuesGuard&) = delete;
                ValuesGuard& operator=(const ValuesGuard&) = delete;
            } values_guard(values_copy_raw, values_copy_on_device);

            // Route points to tiles
            TileAssignment assignment;

            // Location-aware cleanup helpers.
            // GPU mode: assignment arrays and batch local_cell_indices are pool sub-allocations
            // (device memory) — must NOT be passed to free(). Pool reset reclaims them.
            // CPU mode: arrays are malloc'd and must be freed normally.
            auto cleanup_assignment = [](const TileAssignment& a) {
                if (a.location == MemoryLocation::Host) {
                    free(a.cell_indices);
                    free(a.tile_indices);
                    free(a.valid_mask);
                }
                // Device/HostPinned: pool-managed, reclaimed by memory_pool->reset()
            };
            auto cleanup_batches = [](const std::vector<TileBatch>& bs) {
                for (const auto& b : bs) {
                    // CPU path: local_cell_indices is malloc'd in extract_batches_cpu.
                    // GPU path: sub-pointer into pool memory — don't call free().
                    if (b.location == MemoryLocation::Host) {
                        free(b.local_cell_indices);
                    }
                }
            };

            Status s = router->assign(*processing_cloud, assignment);
            if (!s.ok()) {
                if (need_free_indices) free(filtered_indices);
                return s;
            }

            // Sort by tile and cell
            s = router->sort(assignment, values_copy_raw, nullptr, nullptr);
            if (!s.ok()) {
                cleanup_assignment(assignment);
                if (need_free_indices) free(filtered_indices);
                return s;
            }

            // Extract per-tile batches
            std::vector<TileBatch> batches;
            s = router->extract_batches(assignment, values_copy_raw, nullptr, nullptr, batches);
            if (!s.ok()) {
                cleanup_assignment(assignment);
                if (need_free_indices) free(filtered_indices);
                return s;
            }

            // Process each tile batch
            for (const auto& batch : batches) {
                // Acquire tile state
                float* state_ptr = nullptr;
                s = mgr->acquire(batch.tile, reduction.type, &state_ptr);
                if (!s.ok()) {
                    cleanup_batches(batches);
                    cleanup_assignment(assignment);
                    if (need_free_indices) free(filtered_indices);
                    return s;
                }

                // Get actual tile dimensions (may be smaller than configured for edge tiles)
                int col_start, row_start, col_count, row_count;
                config.grid.tile_cell_range(batch.tile, col_start, row_start, col_count, row_count);
                int64_t actual_tile_cells = static_cast<int64_t>(col_count) * row_count;

                // Accumulate points into tile state.
                // In GPU mode, TileManager::acquire() returns a device pointer and handles
                // the H2D upload; TileManager::release() handles the D2H download.
                // So state_ptr is already in the correct memory space for the accumulator.
                s = accumulator->accumulate(reduction.type, batch, state_ptr, actual_tile_cells);
                if (!s.ok()) {
                    mgr->release(batch.tile);
                    cleanup_batches(batches);
                    cleanup_assignment(assignment);
                    if (need_free_indices) free(filtered_indices);
                    return s;
                }

                // Release tile (marks dirty)
                s = mgr->release(batch.tile);
                if (!s.ok()) {
                    cleanup_batches(batches);
                    cleanup_assignment(assignment);
                    if (need_free_indices) free(filtered_indices);
                    return s;
                }
            }

            // Clean up batches and assignment (location-aware)
            cleanup_batches(batches);
            cleanup_assignment(assignment);

            // Reset memory pool for next reduction
            if (memory_pool) {
                memory_pool->reset();
            }
        }

        if (need_free_indices) {
            free(filtered_indices);
        }

        points_processed += filtered_count;
        collections_processed++;

        // Call progress callback
        if (progress_callback) {
            ProgressInfo info;
            info.collections_processed = collections_processed;
            info.collections_total = 0;  // Unknown in streaming mode
            info.points_processed = points_processed;
            info.tiles_active = tile_managers[0]->tiles_in_cache();
            auto now = std::chrono::steady_clock::now();
            info.elapsed_seconds = std::chrono::duration<float>(now - start_time).count();

            bool continue_processing = progress_callback(info);
            if (!continue_processing) {
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: cancelled by user");
            }
        }

        return Status::success();
    }

    // -----------------------------------------------------------------------
    // Hybrid mode: CPU routes/sorts the full cloud, GPU accumulates.
    //
    // Architecture:
    //   - cpu_router routes the input cloud entirely on CPU (no GPU pool).
    //     OpenMP parallelism inside TileRouter is controlled by hybrid_cpu_threads
    //     (set via omp_set_num_threads before the call).
    //   - Resulting host-memory TileBatches are uploaded to a grow-only device
    //     staging buffer via cudaMemcpyAsync, then accumulated via GPU kernels
    //     on the cuda_stream.
    //   - Tile state lives on device (same as pure GPU mode), providing GPU-fast
    //     accumulation with CPU-based routing that avoids GPU pool contention.
    // -----------------------------------------------------------------------
    Status process_cloud_hybrid(const PointCloud& cloud) {
#ifdef PCR_HAS_CUDA
        if (!cpu_router) {
            return Status::error(StatusCode::NotImplemented,
                "pipeline: Hybrid mode requires cpu_router (initialized in Hybrid mode only)");
        }

        size_t num_points = cloud.count();
        if (num_points == 0) return Status::success();

        // For CPU routing, cloud coordinates must be host-accessible.
        // If the cloud is on device, download a host copy first.
        std::unique_ptr<PointCloud> host_cloud;
        const PointCloud* routing_cloud = &cloud;
        if (cloud.location() == MemoryLocation::Device) {
            host_cloud = cloud.to(MemoryLocation::Host);
            if (!host_cloud) {
                return Status::error(StatusCode::CudaError,
                    "pipeline: Hybrid mode failed to download device cloud to host");
            }
            routing_cloud = host_cloud.get();
        }

        // Apply filter (CPU path) if specified
        uint32_t* filtered_indices = nullptr;
        size_t filtered_count = num_points;
        bool need_free_indices = false;

        if (!config.filter.empty()) {
            Status s = filter_points(*routing_cloud, config.filter,
                                     &filtered_indices, &filtered_count);
            if (!s.ok()) return s;
            need_free_indices = true;
        }

        if (filtered_count == 0) {
            if (need_free_indices) free(filtered_indices);
            return Status::success();
        }

        // Determine number of routing threads
        size_t n_threads = config.hybrid_cpu_threads > 0
            ? config.hybrid_cpu_threads
            : std::max((size_t)1, (size_t)std::thread::hardware_concurrency() / 2);
        n_threads = std::min(n_threads, filtered_count);

        // Process each reduction
        for (size_t r = 0; r < config.reductions.size(); ++r) {
            const auto& reduction = config.reductions[r];
            TileManager* mgr = tile_managers[r].get();

            // Get value channel from the routing cloud (host-resident)
            const void* value_data = routing_cloud->channel_data(reduction.value_channel);
            if (!value_data) {
                if (need_free_indices) free(filtered_indices);
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: value channel not found: " + reduction.value_channel);
            }

            const ChannelDesc* desc = routing_cloud->channel(reduction.value_channel);
            if (!desc || desc->dtype != DataType::Float32) {
                if (need_free_indices) free(filtered_indices);
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: value channel must be Float32");
            }

            const float* values = static_cast<const float*>(value_data);

            // Build a gathered host copy of values (filtered subset if needed)
            size_t copy_count = need_free_indices ? filtered_count : num_points;
            float* values_host = static_cast<float*>(malloc(copy_count * sizeof(float)));
            if (!values_host) {
                if (need_free_indices) free(filtered_indices);
                return Status::error(StatusCode::OutOfMemory,
                    "pipeline: Hybrid: failed to allocate host values buffer");
            }

            if (need_free_indices) {
                for (size_t i = 0; i < filtered_count; ++i) {
                    values_host[i] = values[filtered_indices[i]];
                }
            } else {
                std::memcpy(values_host, values, num_points * sizeof(float));
            }

            // ---------------------------------------------------------------
            // Phase 1 (CPU): Route + sort using cpu_router (host memory paths)
            //
            // We route the full cloud at once on CPU.  OpenMP inside TileRouter
            // provides intra-call parallelism controlled by hybrid_cpu_threads
            // (set via omp_set_num_threads at initialization time).
            //
            // Note: n_threads is retained for documentation; actual routing
            // parallelism is handled by OpenMP threads inside cpu_router.
            // ---------------------------------------------------------------
            (void)n_threads;  // routing parallelism via OpenMP inside cpu_router

            // Phase 1: CPU route + sort (OpenMP parallelism inside cpu_router)
            {
#ifdef PCR_HAS_OPENMP
                // Apply hybrid thread count for the routing phase
                if (config.hybrid_cpu_threads > 0) {
                    omp_set_num_threads(static_cast<int>(config.hybrid_cpu_threads));
                }
#endif
                TileAssignment assignment;
                Status s = cpu_router->assign(*routing_cloud, assignment);
                if (!s.ok()) {
                    free(values_host);
                    if (need_free_indices) free(filtered_indices);
                    return s;
                }

                s = cpu_router->sort(assignment, values_host, nullptr, nullptr);
                if (!s.ok()) {
                    free(assignment.cell_indices);
                    free(assignment.tile_indices);
                    free(assignment.valid_mask);
                    free(values_host);
                    if (need_free_indices) free(filtered_indices);
                    return s;
                }

                std::vector<TileBatch> batches;
                s = cpu_router->extract_batches(assignment, values_host,
                                                nullptr, nullptr, batches);
                if (!s.ok()) {
                    free(assignment.cell_indices);
                    free(assignment.tile_indices);
                    free(assignment.valid_mask);
                    free(values_host);
                    if (need_free_indices) free(filtered_indices);
                    return s;
                }

                // ---------------------------------------------------------------
                // Phase 2: GPU accumulation using grow-only device staging buffers.
                // cudaMemcpyAsync on the stream serialises transfers after prior
                // kernels — safe to reuse the same staging buffer every batch.
                // ---------------------------------------------------------------
                uint32_t* d_stage_idx  = nullptr;
                float*    d_stage_vals = nullptr;
                size_t    stage_cap    = 0;

                auto ensure_staging = [&](size_t need) -> bool {
                    if (need <= stage_cap) return true;
                    size_t new_cap = std::max(need, stage_cap * 2);
                    cudaFree(d_stage_idx);
                    cudaFree(d_stage_vals);
                    d_stage_idx = nullptr; d_stage_vals = nullptr; stage_cap = 0;
                    if (cudaMalloc(reinterpret_cast<void**>(&d_stage_idx),
                                   new_cap * sizeof(uint32_t)) != cudaSuccess) return false;
                    if (cudaMalloc(reinterpret_cast<void**>(&d_stage_vals),
                                   new_cap * sizeof(float)) != cudaSuccess) {
                        cudaFree(d_stage_idx); d_stage_idx = nullptr; return false;
                    }
                    stage_cap = new_cap; return true;
                };

                Status accum_error = Status::success();
                for (const auto& batch : batches) {
                    if (!accum_error.ok()) break;

                    int cs, rs, cc, rc;
                    config.grid.tile_cell_range(batch.tile, cs, rs, cc, rc);
                    int64_t actual_tile_cells = static_cast<int64_t>(cc) * rc;

                    if (!ensure_staging(batch.num_points)) {
                        accum_error = Status::error(StatusCode::OutOfMemory,
                            "Hybrid: failed to grow device staging buffer");
                        break;
                    }

                    cudaMemcpyAsync(d_stage_idx, batch.local_cell_indices,
                                    batch.num_points * sizeof(uint32_t),
                                    cudaMemcpyHostToDevice, cuda_stream);
                    cudaMemcpyAsync(d_stage_vals, batch.values,
                                    batch.num_points * sizeof(float),
                                    cudaMemcpyHostToDevice, cuda_stream);

                    TileBatch dev_batch;
                    dev_batch.tile               = batch.tile;
                    dev_batch.local_cell_indices  = d_stage_idx;
                    dev_batch.values              = d_stage_vals;
                    dev_batch.weights             = nullptr;
                    dev_batch.timestamps          = nullptr;
                    dev_batch.num_points          = batch.num_points;
                    dev_batch.location            = MemoryLocation::Device;

                    float* state_ptr = nullptr;
                    s = mgr->acquire(batch.tile, reduction.type, &state_ptr);
                    if (!s.ok()) { accum_error = s; break; }

                    s = accumulator->accumulate(reduction.type, dev_batch,
                                                state_ptr, actual_tile_cells, cuda_stream);
                    if (!s.ok()) { mgr->release(batch.tile); accum_error = s; break; }

                    s = mgr->release(batch.tile);
                    if (!s.ok()) { accum_error = s; break; }
                }

                cudaFree(d_stage_idx);
                cudaFree(d_stage_vals);

                // Free CPU-side batch local_cell_indices (malloc'd by extract_batches_cpu)
                for (const auto& b : batches) {
                    if (b.location == MemoryLocation::Host) free(b.local_cell_indices);
                }
                free(assignment.cell_indices);
                free(assignment.tile_indices);
                free(assignment.valid_mask);

                free(values_host);

                if (!accum_error.ok()) {
                    if (need_free_indices) free(filtered_indices);
                    return accum_error;
                }
            }  // Phase 1+2 scoped block
        }  // for each reduction

        if (need_free_indices) free(filtered_indices);

        points_processed += filtered_count;
        collections_processed++;

        // Progress callback
        if (progress_callback) {
            ProgressInfo info;
            info.collections_processed = collections_processed;
            info.collections_total     = 0;
            info.points_processed      = points_processed;
            info.tiles_active          = tile_managers[0]->tiles_in_cache();
            auto now = std::chrono::steady_clock::now();
            info.elapsed_seconds =
                std::chrono::duration<float>(now - start_time).count();

            if (!progress_callback(info)) {
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: cancelled by user");
            }
        }

        return Status::success();
#else
        // initialize() converts Hybrid→CPU when CUDA is unavailable,
        // so this branch should never be reached in practice.
        (void)cloud;
        return Status::error(StatusCode::NotImplemented,
            "pipeline: Hybrid mode requires a CUDA build");
#endif
    }

    Status finalize_result() {
        // Synchronize CUDA stream before finalization
#ifdef PCR_HAS_CUDA
        if (cuda_stream) {
            cudaError_t err = cudaStreamSynchronize(cuda_stream);
            if (err != cudaSuccess) {
                return Status::error(StatusCode::CudaError,
                    std::string("CUDA stream sync failed during finalization: ") + cudaGetErrorString(err));
            }
        }
#endif

        // Flush all tile managers
        for (auto& mgr : tile_managers) {
            Status s = mgr->flush_all();
            if (!s.ok()) {
                return s;
            }
        }

        // Create output grid
        std::vector<BandDesc> bands;
        for (size_t i = 0; i < config.reductions.size(); ++i) {
            BandDesc band;
            band.name = config.reductions[i].output_band_name.empty()
                ? config.reductions[i].value_channel + "_" + std::to_string(static_cast<int>(config.reductions[i].type))
                : config.reductions[i].output_band_name;
            band.dtype = DataType::Float32;
            band.is_state = false;
            bands.push_back(band);
        }

        result_grid = Grid::create(config.grid.width, config.grid.height, bands, MemoryLocation::Host);
        if (!result_grid) {
            return Status::error(StatusCode::OutOfMemory,
                "pipeline: failed to allocate result grid");
        }

        // For each reduction, merge all tiles into final grid
        for (size_t r = 0; r < config.reductions.size(); ++r) {
            const auto& reduction = config.reductions[r];
            TileManager* mgr = tile_managers[r].get();
            const ReductionInfo* info = get_reduction(reduction.type);

            float* band_data = result_grid->band_f32(static_cast<int>(r));
            if (!band_data) {
                return Status::error(StatusCode::InvalidArgument,
                    "pipeline: failed to get band data");
            }

            // Initialize band to NaN
            size_t grid_cells = config.grid.width * config.grid.height;
            for (size_t i = 0; i < grid_cells; ++i) {
                band_data[i] = NAN;
            }

            // Compute number of tiles
            int tiles_x = (config.grid.width + config.grid.tile_width - 1) / config.grid.tile_width;
            int tiles_y = (config.grid.height + config.grid.tile_height - 1) / config.grid.tile_height;

            // For each tile, load state and finalize
            for (int ty = 0; ty < tiles_y; ++ty) {
                for (int tx = 0; tx < tiles_x; ++tx) {
                    TileIndex tile{ty, tx};

                    // Skip if tile has no state
                    if (!mgr->tile_has_state(tile)) {
                        continue;
                    }

                    // Acquire tile state
                    float* state_ptr = nullptr;
                    Status s = mgr->acquire(tile, reduction.type, &state_ptr);
                    if (!s.ok()) {
                        continue;  // Skip tiles that can't be loaded
                    }

                    // Compute tile bounds in grid
                    int tile_x0 = tx * config.grid.tile_width;
                    int tile_y0 = ty * config.grid.tile_height;
                    int tile_x1 = std::min(tile_x0 + config.grid.tile_width, config.grid.width);
                    int tile_y1 = std::min(tile_y0 + config.grid.tile_height, config.grid.height);

                    int tile_width = tile_x1 - tile_x0;
                    int tile_height = tile_y1 - tile_y0;
                    int tile_cells = tile_width * tile_height;

                    // Use actual tile dimensions for finalization
                    int actual_tile_cells = tile_width * tile_height;

                    // Allocate output buffer for finalized values
                    std::vector<float> finalized(actual_tile_cells, NAN);

                    // In GPU mode, TileManager::acquire() returns a device pointer.
                    // info->finalize() runs on CPU, so download state to host first.
                    std::vector<float> host_state_buf;
                    const float* finalize_state = state_ptr;
#ifdef PCR_HAS_CUDA
                    if (memory_pool && state_ptr) {
                        size_t state_bytes = static_cast<size_t>(info->state_floats)
                                            * actual_tile_cells * sizeof(float);
                        host_state_buf.resize(info->state_floats * actual_tile_cells);
                        cudaMemcpy(host_state_buf.data(), state_ptr, state_bytes,
                                   cudaMemcpyDeviceToHost);
                        finalize_state = host_state_buf.data();
                    }
#endif

                    // Finalize tile state (use actual tile size)
                    s = info->finalize(finalize_state, finalized.data(), actual_tile_cells, nullptr);
                    if (!s.ok()) {
                        mgr->release(tile);
                        continue;
                    }

                    // Copy finalized values to result grid
                    for (int ty_local = 0; ty_local < tile_height; ++ty_local) {
                        for (int tx_local = 0; tx_local < tile_width; ++tx_local) {
                            int grid_x = tile_x0 + tx_local;
                            int grid_y = tile_y0 + ty_local;
                            int grid_idx = grid_y * config.grid.width + grid_x;
                            int tile_idx = ty_local * tile_width + tx_local;  // Use actual tile width, not config
                            band_data[grid_idx] = finalized[tile_idx];
                        }
                    }

                    mgr->release(tile);
                }
            }
        }

        return Status::success();
    }
};

// ---------------------------------------------------------------------------
// Pipeline public API
// ---------------------------------------------------------------------------
Pipeline::~Pipeline() = default;

std::unique_ptr<Pipeline> Pipeline::create(const PipelineConfig& config) {
    auto pipeline = std::unique_ptr<Pipeline>(new Pipeline);
    pipeline->impl_ = std::make_unique<Impl>(config);

    Status s = pipeline->impl_->initialize();
    if (!s.ok()) {
        return nullptr;
    }

    return pipeline;
}

Status Pipeline::validate() const {
    // Validate grid config
    if (impl_->config.grid.width <= 0 || impl_->config.grid.height <= 0) {
        return Status::error(StatusCode::InvalidArgument,
            "pipeline: grid dimensions must be positive");
    }

    if (impl_->config.grid.tile_width <= 0 || impl_->config.grid.tile_height <= 0) {
        return Status::error(StatusCode::InvalidArgument,
            "pipeline: tile dimensions must be positive");
    }

    // Validate reductions
    if (impl_->config.reductions.empty()) {
        return Status::error(StatusCode::InvalidArgument,
            "pipeline: at least one reduction must be specified");
    }

    for (const auto& reduction : impl_->config.reductions) {
        if (reduction.value_channel.empty()) {
            return Status::error(StatusCode::InvalidArgument,
                "pipeline: value_channel must be specified");
        }

        const ReductionInfo* info = get_reduction(reduction.type);
        if (!info) {
            return Status::error(StatusCode::InvalidArgument,
                "pipeline: unknown reduction type");
        }
    }

    return Status::success();
}

Status Pipeline::ingest(const PointCloud& cloud) {
    return impl_->process_cloud(cloud);
}

Status Pipeline::finalize() {
    Status s = impl_->finalize_result();
    if (!s.ok()) {
        return s;
    }

    // Write output if path specified
    if (!impl_->config.output_path.empty()) {
        GeoTiffOptions options;
        options.compress = impl_->config.write_cog ? "DEFLATE" : "NONE";
        options.cloud_optimized = impl_->config.write_cog;

        s = write_geotiff(impl_->config.output_path, *impl_->result_grid,
                         impl_->config.grid, options);
        if (!s.ok()) {
            return s;
        }
    }

    return Status::success();
}

Status Pipeline::run(const std::vector<const PointCloud*>& clouds) {
    for (const auto* cloud : clouds) {
        if (!cloud) {
            return Status::error(StatusCode::InvalidArgument,
                "pipeline: null cloud pointer");
        }

        Status s = ingest(*cloud);
        if (!s.ok()) {
            return s;
        }
    }

    return finalize();
}

void Pipeline::set_progress_callback(ProgressCallback cb) {
    impl_->progress_callback = cb;
}

const Grid* Pipeline::result() const {
    return impl_->result_grid.get();
}

ProgressInfo Pipeline::stats() const {
    ProgressInfo info;
    info.collections_processed = impl_->collections_processed;
    info.collections_total = 0;
    info.points_processed = impl_->points_processed;
    info.tiles_active = impl_->tile_managers.empty() ? 0 : impl_->tile_managers[0]->tiles_in_cache();

    auto now = std::chrono::steady_clock::now();
    info.elapsed_seconds = std::chrono::duration<float>(now - impl_->start_time).count();

    return info;
}

} // namespace pcr
