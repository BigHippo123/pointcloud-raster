# PCR Performance Benchmarking Guide

This directory contains comprehensive benchmarks comparing CPU and GPU performance for the PCR library.

## Prerequisites

- CUDA Toolkit (11.0+)
- CMake (3.18+)
- Python 3.7+ with matplotlib, pandas, seaborn (for visualization)
- NVIDIA GPU with compute capability 7.0+

## Building the Benchmarks

```bash
cd /workspace
mkdir build && cd build

# Configure with CUDA enabled
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DPCR_ENABLE_CUDA=ON \
         -DPCR_BUILD_BENCHMARKS=ON \
         -DPCR_BUILD_TESTS=ON

# Build
make -j$(nproc)
```

## Running Individual Benchmarks

### 1. Memory Pool Benchmark
Tests allocation throughput and reset overhead.

```bash
./bench_memory
```

**Metrics**: allocations/sec, reset overhead, pool vs cudaMalloc speedup

### 2. Sort Benchmark
CUB radix sort performance at various scales.

```bash
./bench_sort
```

**Metrics**: keys/sec, GB/s throughput across 1K to 100M keys

### 3. Accumulate Benchmark
GPU vs CPU accumulation for different reduction types.

```bash
./bench_accumulate
```

**Metrics**: points/sec, speedup for Sum/Max/Min/Count/Average operations

### 4. Full Pipeline Benchmark
End-to-end pipeline performance.

```bash
./bench_tile_pipeline
```

**Metrics**: total points/sec, per-stage breakdown, tiles processed

### 5. Comprehensive CPU vs GPU Comparison
**This is the main benchmark for your performance comparison!**

```bash
./bench_compare_cpu_gpu
```

**Output**: `performance_results.csv`

**Metrics collected**:
- Wall time (ms)
- User time (ms)
- System time (ms)
- Peak RAM usage (KB)
- CPU usage (%)
- GPU time (ms)
- GPU memory used (MB)
- GPU utilization (%)
- Throughput (points/sec, cells/sec)

## Visualizing Results

After running `bench_compare_cpu_gpu`, visualize the results:

```bash
# Install Python dependencies (if needed)
pip install pandas matplotlib seaborn

# Generate visualizations
python visualize_performance.py performance_results.csv
```

**Generated outputs** (in `performance_plots/` directory):

1. **wall_time_comparison.png** - Side-by-side CPU vs GPU wall time
2. **throughput_comparison.png** - Points/sec throughput comparison
3. **gpu_speedup.png** - GPU speedup factors (CPU time / GPU time)
4. **resource_usage.png** - RAM and VRAM usage
5. **utilization.png** - CPU and GPU utilization percentages
6. **summary_table.csv** - Detailed comparison table

## Understanding the Results

### Speedup Calculation
```
Speedup = CPU Wall Time / GPU Wall Time
```
- **>1.0x**: GPU is faster
- **<1.0x**: CPU is faster (unlikely for large datasets)
- **~20-30x**: Expected for well-optimized GPU operations

### CPU Usage %
- **~100%**: Full utilization of 1 CPU core (expected for CPU path)
- **<100%**: Indicates I/O wait or memory bandwidth limitations

### GPU Utilization %
- **~100%**: GPU compute dominates (kernel time ≈ wall time)
- **<50%**: Significant overhead from data transfers or CPU work

### Memory Usage
- **RAM**: Host memory for point data, tile cache
- **VRAM**: Device memory for temporary buffers, point cloud
- Large memory usage is expected for 100M+ point datasets

## Typical Performance Expectations

Based on the implementation:

| Operation | Dataset Size | Expected GPU Speedup |
|-----------|-------------|---------------------|
| Grid Merge | 1M cells | 10-20x |
| Accumulate (Sum) | 10M points | 15-25x |
| Accumulate (Average) | 10M points | 12-20x |
| Full Pipeline | 100M points | 20-30x |

**Factors affecting speedup:**
- **Data transfer overhead**: Smaller datasets see less speedup
- **Atomic contention**: Dense cell coverage reduces GPU efficiency
- **Memory bandwidth**: Large state sizes are memory-bound
- **CPU baseline**: Single-threaded CPU is intentionally slow

## Optimization Notes

If GPU speedup is lower than expected:

1. **Check data transfer time**: Add timing around cudaMemcpy calls
2. **Profile kernel time**: Use `nvprof` or Nsight Compute
3. **Verify GPU utilization**: Use `nvidia-smi` during execution
4. **Check for atomic contention**: Very dense cells may serialize
5. **Memory pool size**: Ensure pool is large enough (check bytes_available)

## Example Session

```bash
# Build
cd /workspace/build
cmake .. -DPCR_ENABLE_CUDA=ON -DPCR_BUILD_BENCHMARKS=ON
make -j8

# Run comprehensive benchmark
./bench_compare_cpu_gpu

# Visualize
cd ../benchmarks
python visualize_performance.py ../build/performance_results.csv

# View results
ls performance_plots/
# wall_time_comparison.png  gpu_speedup.png  ...

# Open images or copy to host for viewing
```

## Customizing Benchmarks

To add custom test cases, edit `bench_compare_cpu_gpu.cu`:

```cpp
// Add to main():
bench_accumulate(csv, YOUR_NUM_POINTS, YOUR_TILE_SIZE);
bench_pipeline(csv, YOUR_NUM_POINTS, YOUR_GRID_SIZE, YOUR_TILE_SIZE);
```

Rebuild and re-run to see your custom workload results.

## CSV Output Format

The `performance_results.csv` contains:

| Column | Description |
|--------|-------------|
| test_name | Test identifier (grid_merge, accumulate_sum, pipeline_e2e) |
| variant | CPU or GPU |
| num_points | Number of points processed |
| num_cells | Number of grid cells |
| wall_time_ms | Total elapsed time |
| user_time_ms | CPU user time |
| system_time_ms | CPU system time |
| peak_ram_kb | Maximum RAM usage |
| cpu_usage_percent | CPU utilization |
| gpu_time_ms | GPU kernel time (CUDA events) |
| gpu_mem_used_mb | GPU memory allocated |
| gpu_utilization_percent | GPU time / wall time |
| points_per_sec | Throughput in points/second |
| cells_per_sec | Throughput in cells/second |

Import this CSV into Excel, Tableau, or your preferred analysis tool for further exploration.

---

**Questions?** Check the main PCR documentation or open an issue on GitHub.
