#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include "pcr/engine/filter.h"
#include "pcr/engine/glyph.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace pcr {

class PointCloud;
class Grid;

// ---------------------------------------------------------------------------
// ReductionSpec — what to reduce and how
// ---------------------------------------------------------------------------
struct ReductionSpec {
    std::string   value_channel;       // channel name to reduce (e.g., "intensity")
    ReductionType type;                // reduction operation

    // Optional — only used by specific ops:
    std::string   weight_channel;      // for WeightedAverage
    std::string   timestamp_channel;   // for MostRecent
    float         percentile = 0.5f;   // for Percentile (0.0 - 1.0)

    // Optional output band name (defaults to "{value_channel}_{type}")
    std::string   output_band_name;

    // Glyph splatting — default GlyphType::Point preserves existing behavior
    GlyphSpec     glyph;
};

// ---------------------------------------------------------------------------
// ExecutionMode — CPU vs GPU execution
// ---------------------------------------------------------------------------
enum class ExecutionMode : uint8_t {
    CPU,    // Force CPU execution
    GPU,    // Force GPU execution (requires CUDA build)
    Auto,   // Automatically choose based on cloud memory location and CUDA availability
    Hybrid  // CPU threads for routing/sorting + GPU for accumulation
};

// ---------------------------------------------------------------------------
// PipelineConfig — full configuration for a processing run
// ---------------------------------------------------------------------------
struct PipelineConfig {
    GridConfig                 grid;
    std::vector<ReductionSpec> reductions;       // one or more reductions to compute
    FilterSpec                 filter;           // optional point filter (applied before reduction)

    // CRS handling
    CRS                        target_crs;       // if set and differs from input, reproject
    bool                       auto_reproject = true;

    // Execution mode
    ExecutionMode exec_mode = ExecutionMode::Auto;

    // Memory budget
    size_t   gpu_memory_budget  = 0;   // 0 = auto-detect (use ~80% of free GPU memory)
    size_t   host_cache_budget  = 0;   // 0 = auto (use ~50% of free host memory)
    size_t   chunk_size         = 0;   // points per chunk, 0 = auto from memory budget

    // GPU configuration (only used if exec_mode == GPU or Auto with GPU-resident clouds)
    size_t   gpu_pool_size_bytes = 512 * 1024 * 1024;  // 512MB default
    int      cuda_device_id = 0;                       // GPU device to use
    bool     use_cuda_streams = true;                  // Enable async CUDA streams
    bool     gpu_fallback_to_cpu = true;               // If true, fall back to CPU on GPU errors
    bool     gpu_require_strict = false;               // If true, fail instead of falling back

    // CPU threading configuration
    size_t   cpu_threads = 0;                          // 0 = use OpenMP default (all available cores)

    // Hybrid mode configuration (Hybrid = CPU routing + GPU accumulation)
    size_t   hybrid_cpu_threads = 0;                   // 0 = hardware_concurrency() / 2

    // State persistence
    std::string state_dir;              // directory for tile state checkpoints
    bool        resume = false;         // if true, load existing state and continue

    // GeoTIFF output
    std::string output_path;            // path for final GeoTIFF
    bool        write_cog = false;      // Cloud-Optimized GeoTIFF
};

// ---------------------------------------------------------------------------
// ProgressCallback — called during pipeline execution
// ---------------------------------------------------------------------------
struct ProgressInfo {
    size_t collections_processed;
    size_t collections_total;       // 0 if unknown (streaming mode)
    size_t points_processed;
    size_t tiles_active;
    float  elapsed_seconds;
};

using ProgressCallback = std::function<bool(const ProgressInfo& info)>;
// Return false from callback to request cancellation.

// ---------------------------------------------------------------------------
// Pipeline — the main entry point for processing
// ---------------------------------------------------------------------------
class Pipeline {
public:
    ~Pipeline();

    static std::unique_ptr<Pipeline> create(const PipelineConfig& config);

    /// Validate configuration before running.
    Status validate() const;

    // -- Batch mode: provide all collections up front ----------------------

    /// Process a single PointCloud (one collection).
    /// Can be called repeatedly to add collections incrementally.
    /// Points can be on Host or Device.
    Status ingest(const PointCloud& cloud);

    /// Signal that all collections have been ingested.
    /// Finalizes reduction state and writes output.
    Status finalize();

    // -- Convenience: process a list of clouds in one call -----------------

    Status run(const std::vector<const PointCloud*>& clouds);

    // -- Progress and cancellation -----------------------------------------

    void set_progress_callback(ProgressCallback cb);

    // -- Access results after finalize() -----------------------------------

    /// Get the finalized grid (full assembly of all tiles).
    /// Only valid after finalize(). Returns nullptr if not yet finalized.
    const Grid* result() const;

    /// Get stats about the completed run.
    ProgressInfo stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcr
