#include "pcr/engine/tile_manager.h"
#include "pcr/core/grid.h"
#include "pcr/ops/reduction_registry.h"
#include "pcr/io/tile_state_io.h"
#include <map>
#include <list>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// TileState - represents a tile in the cache
// ---------------------------------------------------------------------------
struct TileState {
    TileIndex tile;
    float*    state_ptr = nullptr;   // host memory buffer (always allocated for cache)
    float*    device_ptr = nullptr;  // device memory buffer (if using GPU)
    size_t    size_bytes = 0;
    bool      dirty = false;         // needs flush to disk
    bool      pinned = false;        // currently acquired (don't evict)
    ReductionType reduction_type;
    MemoryLocation location = MemoryLocation::Host;

    ~TileState() {
        if (state_ptr) {
            free(state_ptr);
        }
#ifdef PCR_HAS_CUDA
        if (device_ptr) {
            cudaFree(device_ptr);
        }
#endif
    }
};

// ---------------------------------------------------------------------------
// TileManager::Impl
// ---------------------------------------------------------------------------
struct TileManager::Impl {
    TileManagerConfig config;

    // LRU cache: map from TileIndex to TileState
    std::map<TileIndex, std::unique_ptr<TileState>> cache;

    // LRU ordering: most recently used at front, least recently used at back
    std::list<TileIndex> lru_list;

    // Cache position lookup for O(1) access
    std::map<TileIndex, std::list<TileIndex>::iterator> lru_position;

    // Stats
    size_t cache_hits_count = 0;
    size_t cache_misses_count = 0;
    size_t current_cache_bytes = 0;

    int tile_cells = 0;  // computed from grid_config

    Impl(const TileManagerConfig& cfg) : config(cfg) {
        tile_cells = config.grid_config.tile_width * config.grid_config.tile_height;
    }

    // Touch tile in LRU (move to front)
    void touch(TileIndex tile) {
        auto it = lru_position.find(tile);
        if (it != lru_position.end()) {
            lru_list.erase(it->second);
        }
        lru_list.push_front(tile);
        lru_position[tile] = lru_list.begin();
    }

    // Evict least recently used tile (if not pinned)
    Status evict_lru() {
        // Find LRU unpinned tile
        for (auto it = lru_list.rbegin(); it != lru_list.rend(); ++it) {
            TileIndex tile = *it;
            auto tile_it = cache.find(tile);
            if (tile_it != cache.end() && !tile_it->second->pinned) {
                // Flush if dirty
                if (tile_it->second->dirty) {
                    Status s = flush_tile(tile);
                    if (!s.ok()) {
                        return s;
                    }
                }

                // Remove from cache
                current_cache_bytes -= tile_it->second->size_bytes;
                lru_position.erase(tile);
                lru_list.erase(std::next(it).base());
                cache.erase(tile_it);
                return Status::success();
            }
        }

        return Status::error(StatusCode::OutOfMemory, "tile_manager: all tiles are pinned, cannot evict");
    }

    // Flush single tile to disk
    Status flush_tile(TileIndex tile) {
        auto it = cache.find(tile);
        if (it == cache.end()) {
            return Status::error(StatusCode::InvalidArgument, "tile_manager: tile not in cache");
        }

        TileState* ts = it->second.get();
        if (!ts->dirty) {
            return Status::success();  // Nothing to flush
        }

        std::string path = tile_state_filename(config.state_dir, tile);

        // Get actual tile dimensions for writing metadata
        int col_start, row_start, col_count, row_count;
        config.grid_config.tile_cell_range(tile, col_start, row_start, col_count, row_count);

        Status s = write_tile_state(
            path,
            tile,
            col_count,
            row_count,
            config.state_floats,
            ts->reduction_type,
            ts->state_ptr
        );

        if (s.ok()) {
            ts->dirty = false;
        }

        return s;
    }

    // Check if tile file exists on disk
    bool tile_file_exists(TileIndex tile) const {
        std::string path = tile_state_filename(config.state_dir, tile);
        struct stat buffer;
        return (stat(path.c_str(), &buffer) == 0);
    }

    // Create state directory if it doesn't exist
    Status ensure_state_dir() {
        struct stat st;
        if (stat(config.state_dir.c_str(), &st) != 0) {
            // Directory doesn't exist, create it
#ifdef _WIN32
            if (mkdir(config.state_dir.c_str()) != 0) {
#else
            if (mkdir(config.state_dir.c_str(), 0755) != 0) {
#endif
                return Status::error(StatusCode::IoError,
                    "tile_manager: failed to create state directory: " + config.state_dir);
            }
        }
        return Status::success();
    }
};

// ---------------------------------------------------------------------------
// TileManager public API
// ---------------------------------------------------------------------------
TileManager::~TileManager() = default;

std::unique_ptr<TileManager> TileManager::create(const TileManagerConfig& config) {
    auto mgr = std::unique_ptr<TileManager>(new TileManager);
    mgr->impl_ = std::make_unique<Impl>(config);

    // Ensure state directory exists
    Status s = mgr->impl_->ensure_state_dir();
    if (!s.ok()) {
        return nullptr;
    }

    return mgr;
}

Status TileManager::acquire(TileIndex tile, ReductionType type, float** out_host_ptr) {
    if (!out_host_ptr) {
        return Status::error(StatusCode::InvalidArgument, "tile_manager: out_host_ptr is null");
    }

    // Check if tile is already in cache
    auto it = impl_->cache.find(tile);
    if (it != impl_->cache.end()) {
        // Cache hit
        impl_->cache_hits_count++;
        impl_->touch(tile);
        TileState* ts = it->second.get();
        ts->pinned = true;

#ifdef PCR_HAS_CUDA
        // If using GPU but device memory not allocated yet, allocate and transfer
        if (impl_->config.memory_location == MemoryLocation::Device && !ts->device_ptr) {
            cudaError_t err = cudaMalloc(&ts->device_ptr, ts->size_bytes);
            if (err != cudaSuccess) {
                return Status::error(StatusCode::CudaError,
                    std::string("tile_manager: failed to allocate device state: ") +
                    cudaGetErrorString(err));
            }

            // Transfer host state to device
            cudaStream_t stream = static_cast<cudaStream_t>(impl_->config.cuda_stream);
            if (stream) {
                err = cudaMemcpyAsync(ts->device_ptr, ts->state_ptr, ts->size_bytes,
                                     cudaMemcpyHostToDevice, stream);
            } else {
                err = cudaMemcpy(ts->device_ptr, ts->state_ptr, ts->size_bytes,
                                cudaMemcpyHostToDevice);
            }

            if (err != cudaSuccess) {
                return Status::error(StatusCode::CudaError,
                    std::string("tile_manager: failed to transfer state to device: ") +
                    cudaGetErrorString(err));
            }

            ts->location = MemoryLocation::Device;
        }

        // Return appropriate pointer
        if (impl_->config.memory_location == MemoryLocation::Device) {
            *out_host_ptr = ts->device_ptr;
        } else {
            *out_host_ptr = ts->state_ptr;
        }
#else
        *out_host_ptr = ts->state_ptr;
#endif

        return Status::success();
    }

    // Cache miss
    impl_->cache_misses_count++;

    // Get actual tile dimensions for this specific tile (may be smaller for edge tiles)
    int col_start, row_start, col_count, row_count;
    impl_->config.grid_config.tile_cell_range(tile, col_start, row_start, col_count, row_count);
    int64_t actual_tile_cells = static_cast<int64_t>(col_count) * row_count;

    // Calculate required memory using actual tile size
    size_t state_size = impl_->config.state_floats * actual_tile_cells;
    size_t size_bytes = state_size * sizeof(float);

    // Evict tiles if necessary to make room
    while (impl_->current_cache_bytes + size_bytes > impl_->config.cache_size_bytes) {
        Status s = impl_->evict_lru();
        if (!s.ok()) {
            return s;
        }
    }

    // Allocate new tile state
    auto ts = std::make_unique<TileState>();
    ts->tile = tile;
    ts->size_bytes = size_bytes;
    ts->state_ptr = static_cast<float*>(malloc(size_bytes));
    ts->reduction_type = type;
    ts->pinned = true;
    ts->dirty = false;

    if (!ts->state_ptr) {
        return Status::error(StatusCode::OutOfMemory, "tile_manager: failed to allocate tile state");
    }

    // Try to load from disk - but validate dimensions BEFORE reading data
    std::string path = tile_state_filename(impl_->config.state_dir, tile);

    TileIndex loaded_tile;
    int loaded_cols, loaded_rows, loaded_state_floats;
    ReductionType loaded_type;

    // Step 1: Read header only to check dimensions (safe - no buffer overflow risk)
    Status s = read_tile_state_header(path, loaded_tile, loaded_cols, loaded_rows,
                                       loaded_state_floats, loaded_type);

    bool should_initialize = true;

    if (s.ok()) {
        // Header read successfully - now validate dimensions BEFORE reading data
        if (loaded_cols == col_count &&
            loaded_rows == row_count &&
            loaded_state_floats == impl_->config.state_floats &&
            loaded_tile.row == tile.row &&
            loaded_tile.col == tile.col) {
            // Dimensions match - safe to read full data
            s = read_tile_state(path, loaded_tile, loaded_cols, loaded_rows,
                               loaded_state_floats, loaded_type, ts->state_ptr);

            if (s.ok()) {
                should_initialize = false;  // Loaded from disk successfully
            }
            // If read failed, fall through to initialization
        }
        // If dimensions mismatch, discard stale file and initialize fresh
    }
    // If header read failed (file doesn't exist or is corrupt), initialize fresh

    if (should_initialize) {
        // File doesn't exist, is corrupt, or has wrong dimensions - initialize to identity
        const ReductionInfo* info = get_reduction(type);
        if (!info) {
            return Status::error(StatusCode::InvalidArgument,
                "tile_manager: unknown reduction type");
        }

        s = info->init_state(ts->state_ptr, actual_tile_cells, nullptr);
        if (!s.ok()) {
            return s;
        }

        // Mark as dirty since we initialized it
        ts->dirty = true;
    }

    // Allocate on device and transfer if using GPU
    ts->location = impl_->config.memory_location;

#ifdef PCR_HAS_CUDA
    if (impl_->config.memory_location == MemoryLocation::Device ||
        impl_->config.memory_location == MemoryLocation::HostPinned) {

        // Allocate device memory
        if (impl_->config.memory_location == MemoryLocation::Device) {
            cudaError_t err = cudaMalloc(&ts->device_ptr, size_bytes);
            if (err != cudaSuccess) {
                return Status::error(StatusCode::CudaError,
                    std::string("tile_manager: failed to allocate device state: ") +
                    cudaGetErrorString(err));
            }

            // Transfer host state to device
            cudaStream_t stream = static_cast<cudaStream_t>(impl_->config.cuda_stream);
            if (stream) {
                err = cudaMemcpyAsync(ts->device_ptr, ts->state_ptr, size_bytes,
                                     cudaMemcpyHostToDevice, stream);
            } else {
                err = cudaMemcpy(ts->device_ptr, ts->state_ptr, size_bytes,
                                cudaMemcpyHostToDevice);
            }

            if (err != cudaSuccess) {
                return Status::error(StatusCode::CudaError,
                    std::string("tile_manager: failed to transfer state to device: ") +
                    cudaGetErrorString(err));
            }
        }
    }
#endif

    // Add to cache
    impl_->current_cache_bytes += size_bytes;
    impl_->touch(tile);

    // Return appropriate pointer based on memory location
    if (impl_->config.memory_location == MemoryLocation::Device) {
#ifdef PCR_HAS_CUDA
        *out_host_ptr = ts->device_ptr;  // Return device pointer
#else
        return Status::error(StatusCode::NotImplemented, "tile_manager: Device memory requires CUDA");
#endif
    } else {
        *out_host_ptr = ts->state_ptr;  // Return host pointer
    }

    impl_->cache[tile] = std::move(ts);

    return Status::success();
}

Status TileManager::release(TileIndex tile) {
    auto it = impl_->cache.find(tile);
    if (it == impl_->cache.end()) {
        return Status::error(StatusCode::InvalidArgument, "tile_manager: tile not in cache");
    }

    TileState* ts = it->second.get();

    // Transfer device state back to host before releasing (for flushing to disk later)
#ifdef PCR_HAS_CUDA
    if (ts->location == MemoryLocation::Device && ts->device_ptr) {
        cudaStream_t stream = static_cast<cudaStream_t>(impl_->config.cuda_stream);
        cudaError_t err;

        if (stream) {
            err = cudaMemcpyAsync(ts->state_ptr, ts->device_ptr, ts->size_bytes,
                                 cudaMemcpyDeviceToHost, stream);
            // Synchronize to ensure transfer completes before release
            cudaStreamSynchronize(stream);
        } else {
            err = cudaMemcpy(ts->state_ptr, ts->device_ptr, ts->size_bytes,
                            cudaMemcpyDeviceToHost);
        }

        if (err != cudaSuccess) {
            return Status::error(StatusCode::CudaError,
                std::string("tile_manager: failed to transfer state from device: ") +
                cudaGetErrorString(err));
        }
    }
#endif

    // Unpin and mark dirty
    ts->pinned = false;
    ts->dirty = true;

    return Status::success();
}

Status TileManager::flush_all() {
    for (auto& kv : impl_->cache) {
        if (kv.second->dirty) {
            Status s = impl_->flush_tile(kv.first);
            if (!s.ok()) {
                return s;
            }
        }
    }
    return Status::success();
}

void TileManager::clear_cache() {
    impl_->cache.clear();
    impl_->lru_list.clear();
    impl_->lru_position.clear();
    impl_->current_cache_bytes = 0;
    impl_->cache_hits_count = 0;
    impl_->cache_misses_count = 0;
}

bool TileManager::tile_has_state(TileIndex tile) const {
    // Check cache first
    if (impl_->cache.find(tile) != impl_->cache.end()) {
        return true;
    }
    // Check disk
    return impl_->tile_file_exists(tile);
}

Status TileManager::reset() {
    // Clear cache without flushing
    clear_cache();

    // Delete all state files
    // Note: This is a simple implementation that relies on the user to clean up
    // A more robust implementation would enumerate and delete files
    return Status::success();
}

size_t TileManager::cache_hits() const {
    return impl_->cache_hits_count;
}

size_t TileManager::cache_misses() const {
    return impl_->cache_misses_count;
}

size_t TileManager::tiles_on_disk() const {
    // This would require enumerating the state directory
    // For now, return 0 as a placeholder
    return 0;
}

size_t TileManager::tiles_in_cache() const {
    return impl_->cache.size();
}

} // namespace pcr
