#include "pcr/engine/accumulator.h"
#include "pcr/engine/grid_merge.h"
#include "pcr/engine/filter.h"
#include "pcr/engine/tile_router.h"
#include "pcr/engine/memory_pool.h"
#include "pcr/engine/pipeline.h"
#include "pcr/core/point_cloud.h"
#include "pcr/core/grid_config.h"
#include "pcr/ops/reduction_registry.h"
#include <cuda_runtime.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <unistd.h>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <sstream>

using namespace pcr;

// ---------------------------------------------------------------------------
// Performance Metrics Collection
// ---------------------------------------------------------------------------

struct PerfMetrics {
    double wall_time_ms;
    double user_time_ms;
    double system_time_ms;
    size_t peak_ram_kb;
    double cpu_usage_percent;

    // GPU-specific
    double gpu_time_ms;
    size_t gpu_mem_used_mb;
    double gpu_utilization_percent;

    // Throughput
    double points_per_sec;
    double cells_per_sec;
};

struct ResourceSnapshot {
    struct rusage usage;
    std::chrono::steady_clock::time_point wall_time;
    cudaEvent_t cuda_start;
    cudaEvent_t cuda_stop;
    size_t gpu_mem_before;
    size_t gpu_mem_after;
};

class PerformanceProfiler {
public:
    void start(bool use_gpu = false) {
        getrusage(RUSAGE_SELF, &snap_.usage);
        snap_.wall_time = std::chrono::steady_clock::now();

        if (use_gpu) {
            cudaMemGetInfo(&snap_.gpu_mem_before, nullptr);
            cudaEventCreate(&snap_.cuda_start);
            cudaEventCreate(&snap_.cuda_stop);
            cudaEventRecord(snap_.cuda_start);
        }
    }

    PerfMetrics stop(bool use_gpu = false, size_t num_points = 0, size_t num_cells = 0) {
        PerfMetrics m = {};

        // Wall time
        auto now = std::chrono::steady_clock::now();
        m.wall_time_ms = std::chrono::duration<double, std::milli>(now - snap_.wall_time).count();

        // CPU time
        struct rusage usage_end;
        getrusage(RUSAGE_SELF, &usage_end);

        double user_sec = (usage_end.ru_utime.tv_sec - snap_.usage.ru_utime.tv_sec) +
                         (usage_end.ru_utime.tv_usec - snap_.usage.ru_utime.tv_usec) / 1e6;
        double sys_sec = (usage_end.ru_stime.tv_sec - snap_.usage.ru_stime.tv_sec) +
                        (usage_end.ru_stime.tv_usec - snap_.usage.ru_stime.tv_usec) / 1e6;

        m.user_time_ms = user_sec * 1000.0;
        m.system_time_ms = sys_sec * 1000.0;
        m.peak_ram_kb = usage_end.ru_maxrss;
        m.cpu_usage_percent = ((user_sec + sys_sec) / (m.wall_time_ms / 1000.0)) * 100.0;

        // GPU metrics
        if (use_gpu) {
            cudaEventRecord(snap_.cuda_stop);
            cudaEventSynchronize(snap_.cuda_stop);

            float cuda_ms = 0;
            cudaEventElapsedTime(&cuda_ms, snap_.cuda_start, snap_.cuda_stop);
            m.gpu_time_ms = cuda_ms;

            cudaMemGetInfo(&snap_.gpu_mem_after, nullptr);
            size_t mem_used = snap_.gpu_mem_before - snap_.gpu_mem_after;
            m.gpu_mem_used_mb = mem_used / (1024 * 1024);

            // GPU utilization estimate (rough)
            m.gpu_utilization_percent = (m.gpu_time_ms / m.wall_time_ms) * 100.0;

            cudaEventDestroy(snap_.cuda_start);
            cudaEventDestroy(snap_.cuda_stop);
        }

        // Throughput
        if (num_points > 0) {
            m.points_per_sec = num_points / (m.wall_time_ms / 1000.0);
        }
        if (num_cells > 0) {
            m.cells_per_sec = num_cells / (m.wall_time_ms / 1000.0);
        }

        return m;
    }

private:
    ResourceSnapshot snap_;
};

// ---------------------------------------------------------------------------
// CSV Logger
// ---------------------------------------------------------------------------

class CSVLogger {
public:
    CSVLogger(const std::string& filename) : filename_(filename) {
        file_.open(filename);
        write_header();
    }

    ~CSVLogger() {
        if (file_.is_open()) file_.close();
    }

    void log(const std::string& test_name, const std::string& variant,
             size_t num_points, size_t num_cells, const PerfMetrics& m) {
        file_ << test_name << ","
              << variant << ","
              << num_points << ","
              << num_cells << ","
              << std::fixed << std::setprecision(3)
              << m.wall_time_ms << ","
              << m.user_time_ms << ","
              << m.system_time_ms << ","
              << m.peak_ram_kb << ","
              << m.cpu_usage_percent << ","
              << m.gpu_time_ms << ","
              << m.gpu_mem_used_mb << ","
              << m.gpu_utilization_percent << ","
              << std::scientific
              << m.points_per_sec << ","
              << m.cells_per_sec << "\n";
        file_.flush();
    }

private:
    void write_header() {
        file_ << "test_name,variant,num_points,num_cells,"
              << "wall_time_ms,user_time_ms,system_time_ms,peak_ram_kb,cpu_usage_percent,"
              << "gpu_time_ms,gpu_mem_used_mb,gpu_utilization_percent,"
              << "points_per_sec,cells_per_sec\n";
    }

    std::string filename_;
    std::ofstream file_;
};

// ---------------------------------------------------------------------------
// Test Data Generation
// ---------------------------------------------------------------------------

std::unique_ptr<PointCloud> generate_point_cloud(size_t n, const BBox& bounds, unsigned seed) {
    auto cloud = PointCloud::create(n, MemoryLocation::Host);
    cloud->resize(n);
    cloud->add_channel("intensity", DataType::Float32);
    cloud->add_channel("classification", DataType::Float32);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist_x(bounds.min_x, bounds.max_x);
    std::uniform_real_distribution<double> dist_y(bounds.min_y, bounds.max_y);
    std::uniform_real_distribution<float> dist_val(0.0f, 100.0f);
    std::uniform_int_distribution<int> dist_class(0, 5);

    for (size_t i = 0; i < n; ++i) {
        cloud->x()[i] = dist_x(rng);
        cloud->y()[i] = dist_y(rng);
        cloud->channel_f32("intensity")[i] = dist_val(rng);
        cloud->channel_f32("classification")[i] = static_cast<float>(dist_class(rng));
    }

    return cloud;
}

// ---------------------------------------------------------------------------
// Benchmark: Grid Merge Operations
// ---------------------------------------------------------------------------

void bench_grid_merge(CSVLogger& csv, size_t num_cells) {
    printf("=== GridMerge Benchmark (N=%zu cells) ===\n", num_cells);

    // CPU version
    {
        const ReductionInfo* info = get_reduction(ReductionType::Sum);
        std::vector<float> state_a(num_cells, 0.0f);
        std::vector<float> state_b(num_cells, 0.0f);

        for (size_t i = 0; i < num_cells; ++i) {
            state_a[i] = static_cast<float>(i);
            state_b[i] = static_cast<float>(i * 2);
        }

        PerformanceProfiler prof;
        prof.start(false);
        info->merge_state(state_a.data(), state_b.data(), num_cells, nullptr);
        PerfMetrics m = prof.stop(false, 0, num_cells);

        csv.log("grid_merge", "CPU", 0, num_cells, m);
        printf("  CPU: %.3f ms, %.2f M cells/sec, RAM: %zu KB\n",
               m.wall_time_ms, m.cells_per_sec / 1e6, m.peak_ram_kb);
    }

    // GPU version
    {
        std::vector<float> h_state_a(num_cells, 0.0f);
        std::vector<float> h_state_b(num_cells, 0.0f);

        for (size_t i = 0; i < num_cells; ++i) {
            h_state_a[i] = static_cast<float>(i);
            h_state_b[i] = static_cast<float>(i * 2);
        }

        float *d_state_a, *d_state_b;
        cudaMalloc(&d_state_a, num_cells * sizeof(float));
        cudaMalloc(&d_state_b, num_cells * sizeof(float));
        cudaMemcpy(d_state_a, h_state_a.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_state_b, h_state_b.data(), num_cells * sizeof(float), cudaMemcpyHostToDevice);

        PerformanceProfiler prof;
        prof.start(true);
        merge_tile_state(ReductionType::Sum, d_state_a, d_state_b, num_cells);
        cudaDeviceSynchronize();
        PerfMetrics m = prof.stop(true, 0, num_cells);

        csv.log("grid_merge", "GPU", 0, num_cells, m);
        printf("  GPU: %.3f ms, %.2f M cells/sec, VRAM: %zu MB, Speedup: %.1fx\n",
               m.wall_time_ms, m.cells_per_sec / 1e6, m.gpu_mem_used_mb,
               m.cells_per_sec / (num_cells / (m.wall_time_ms / 1000.0)));

        cudaFree(d_state_a);
        cudaFree(d_state_b);
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Benchmark: Accumulate Operations
// ---------------------------------------------------------------------------

void bench_accumulate(CSVLogger& csv, size_t num_points, size_t tile_cells) {
    printf("=== Accumulate Benchmark (N=%zu points, %zu cells) ===\n", num_points, tile_cells);

    // Generate sorted cell indices
    std::vector<uint32_t> h_cells(num_points);
    std::vector<float> h_values(num_points);
    std::mt19937 rng(42);
    for (size_t i = 0; i < num_points; ++i) {
        h_cells[i] = rng() % tile_cells;
        h_values[i] = static_cast<float>(rng() % 1000) / 10.0f;
    }
    std::sort(h_cells.begin(), h_cells.end());

    // CPU version
    {
        const ReductionInfo* info = get_reduction(ReductionType::Sum);
        std::vector<float> state(tile_cells, 0.0f);

        PerformanceProfiler prof;
        prof.start(false);
        info->init_state(state.data(), tile_cells, nullptr);
        info->accumulate(h_cells.data(), h_values.data(), state.data(), num_points, tile_cells, nullptr);
        PerfMetrics m = prof.stop(false, num_points, tile_cells);

        csv.log("accumulate_sum", "CPU", num_points, tile_cells, m);
        printf("  CPU: %.3f ms, %.2f M pts/sec\n", m.wall_time_ms, m.points_per_sec / 1e6);
    }

    // GPU version
    {
        auto pool = MemoryPool::create(512 * 1024 * 1024);
        auto acc = Accumulator::create(pool.get());

        uint32_t *d_cells;
        float *d_values, *d_state;
        cudaMalloc(&d_cells, num_points * sizeof(uint32_t));
        cudaMalloc(&d_values, num_points * sizeof(float));
        cudaMalloc(&d_state, tile_cells * sizeof(float));
        cudaMemcpy(d_cells, h_cells.data(), num_points * sizeof(uint32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, h_values.data(), num_points * sizeof(float), cudaMemcpyHostToDevice);

        TileBatch batch;
        batch.tile = {0, 0};
        batch.local_cell_indices = d_cells;
        batch.values = d_values;
        batch.num_points = num_points;

        PerformanceProfiler prof;
        prof.start(true);
        init_tile_state(ReductionType::Sum, d_state, tile_cells);
        acc->accumulate(ReductionType::Sum, batch, d_state, tile_cells);
        cudaDeviceSynchronize();
        PerfMetrics m = prof.stop(true, num_points, tile_cells);

        csv.log("accumulate_sum", "GPU", num_points, tile_cells, m);
        double cpu_rate = num_points / (m.wall_time_ms / 1000.0);  // Approx from previous
        printf("  GPU: %.3f ms, %.2f M pts/sec, Speedup: ~%.1fx\n",
               m.wall_time_ms, m.points_per_sec / 1e6, m.points_per_sec / cpu_rate);

        cudaFree(d_cells);
        cudaFree(d_values);
        cudaFree(d_state);
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// Benchmark: End-to-End Pipeline
// ---------------------------------------------------------------------------

void bench_pipeline(CSVLogger& csv, size_t num_points, int grid_size, int tile_size) {
    printf("=== Pipeline Benchmark (N=%zu points, grid=%dx%d, tile=%d) ===\n",
           num_points, grid_size, grid_size, tile_size);

    GridConfig grid;
    grid.bounds = {0.0, 0.0, static_cast<double>(grid_size), static_cast<double>(grid_size)};
    grid.cell_size_x = 1.0;
    grid.cell_size_y = -1.0;
    grid.tile_width = tile_size;
    grid.tile_height = tile_size;
    grid.compute_dimensions();

    auto cloud = generate_point_cloud(num_points, grid.bounds, 42);

    PipelineConfig config;
    config.grid = grid;
    config.reductions.push_back({"intensity", ReductionType::Sum});
    config.reductions.push_back({"intensity", ReductionType::Average});
    config.state_dir = "/tmp/pcr_bench_pipeline";

    PerformanceProfiler prof;
    prof.start(true);  // GPU path enabled

    auto pipeline = Pipeline::create(config);
    if (!pipeline) {
        printf("  ERROR: Failed to create pipeline\n");
        return;
    }

    Status s = pipeline->ingest(*cloud);
    if (!s.ok()) {
        printf("  ERROR ingest: %s\n", s.message.c_str());
        return;
    }

    s = pipeline->finalize();
    if (!s.ok()) {
        printf("  ERROR finalize: %s\n", s.message.c_str());
        return;
    }

    PerfMetrics m = prof.stop(true, num_points, grid_size * grid_size);
    csv.log("pipeline_e2e", "GPU", num_points, grid_size * grid_size, m);

    printf("  Wall: %.1f ms, GPU: %.1f ms, %.2f M pts/sec\n",
           m.wall_time_ms, m.gpu_time_ms, m.points_per_sec / 1e6);
    printf("  RAM: %zu KB, VRAM: %zu MB, CPU: %.1f%%, GPU util: %.1f%%\n",
           m.peak_ram_kb, m.gpu_mem_used_mb, m.cpu_usage_percent, m.gpu_utilization_percent);
    printf("\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    // Print system info
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("===========================================\n");
    printf("CPU vs GPU Performance Comparison\n");
    printf("===========================================\n");
    printf("GPU: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Memory: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    // printf("  Clock: %.0f MHz\n", prop.clockRate / 1000.0); // clockRate deprecated in CUDA 12+
    printf("\n");

    CSVLogger csv("performance_results.csv");

    // Grid merge tests
    bench_grid_merge(csv, 1024 * 1024);     // 1M cells
    bench_grid_merge(csv, 4096 * 4096);     // 16M cells

    // Accumulate tests
    bench_accumulate(csv, 1000000, 1024 * 1024);      // 1M points
    bench_accumulate(csv, 10000000, 1024 * 1024);     // 10M points
    bench_accumulate(csv, 100000000, 4096 * 4096);    // 100M points

    // End-to-end pipeline tests
    bench_pipeline(csv, 1000000, 2048, 512);          // 1M points, 4M cells
    bench_pipeline(csv, 10000000, 4096, 1024);        // 10M points, 16M cells
    bench_pipeline(csv, 100000000, 8192, 1024);       // 100M points, 67M cells

    printf("===========================================\n");
    printf("Results saved to: performance_results.csv\n");
    printf("===========================================\n");

    return 0;
}
