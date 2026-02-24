# PCR Benchmarking Quick Start

## TL;DR - Run Everything

```bash
# 1. Build with CUDA
cd /workspace
mkdir build && cd build
cmake .. -DPCR_ENABLE_CUDA=ON -DPCR_BUILD_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 2. Run all benchmarks (automated)
../benchmarks/run_all_benchmarks.sh

# 3. Results will be in: benchmark_results_YYYYMMDD_HHMMSS/
```

## What You'll Get

### 📊 Performance Data Files

- **performance_results.csv** - Raw metrics (wall time, RAM, VRAM, CPU/GPU usage, throughput)
- **summary_table.csv** - Clean comparison table with speedup factors
- **bench_*.txt** - Detailed output from each benchmark

### 📈 Visualizations (PNG images)

- **wall_time_comparison.png** - CPU vs GPU execution time
- **throughput_comparison.png** - Points/sec and cells/sec
- **gpu_speedup.png** - Speedup factors (how much faster GPU is)
- **resource_usage.png** - RAM and VRAM consumption
- **utilization.png** - CPU and GPU usage percentages

## Manual Run (for custom tests)

```bash
cd build

# Comprehensive comparison (saves CSV)
./bench_compare_cpu_gpu

# Individual benchmarks
./bench_memory          # Memory pool performance
./bench_sort            # CUB sort throughput
./bench_accumulate      # Reduction operations
./bench_tile_pipeline   # End-to-end pipeline

# Visualize results
python ../benchmarks/visualize_performance.py performance_results.csv
```

## Interpreting Results

### Key Metrics to Watch

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **GPU Speedup** | How many times faster GPU is vs CPU | 20-30x for large datasets |
| **Points/sec** | Processing throughput | 50M+ pts/sec on modern GPUs |
| **GPU Utilization** | % of time GPU is computing | >80% means compute-bound |
| **VRAM Usage** | GPU memory consumed | <50% of total GPU memory |
| **CPU Usage** | Host CPU load | ~100% for CPU path, <50% for GPU path |

### Expected Performance

**Grid Merge (1M cells)**
- CPU: ~500ms
- GPU: ~25ms
- **Speedup: ~20x**

**Accumulate Sum (10M points)**
- CPU: ~800ms
- GPU: ~35ms
- **Speedup: ~23x**

**Full Pipeline (100M points, 4096x4096 grid)**
- CPU: ~60 seconds
- GPU: ~2 seconds
- **Speedup: ~30x**

## Performance Comparison Example

After running benchmarks, your `summary_table.csv` will look like:

```
Test,Points/Cells,CPU Time (ms),GPU Time (ms),Speedup,CPU RAM (MB),GPU VRAM (MB),GPU Throughput (Mpts/s)
accumulate_sum,"1,000,000",850.23,38.45,22.11x,45.2,128.5,26.01
accumulate_sum,"10,000,000",8234.56,348.91,23.59x,452.1,1024.3,28.65
pipeline_e2e,"100,000,000",58934.21,1987.34,29.66x,4832.5,2048.7,50.32
```

## Visualizations Preview

The generated plots will show:

1. **Bar charts** comparing CPU (blue) vs GPU (orange) execution times
2. **Speedup chart** showing GPU gains (green bars = faster)
3. **Resource usage** showing memory footprint
4. **Utilization** showing how efficiently hardware is used

## Troubleshooting

**No GPU detected:**
```bash
nvidia-smi  # Verify GPU is visible
# If not, check CUDA drivers
```

**Benchmarks fail to compile:**
```bash
# Ensure CUDA is enabled
cmake .. -DPCR_ENABLE_CUDA=ON -DPCR_BUILD_BENCHMARKS=ON
```

**Python visualization fails:**
```bash
pip install pandas matplotlib seaborn
```

**Out of memory errors:**
- Reduce test sizes in `bench_compare_cpu_gpu.cu`
- Check GPU memory with `nvidia-smi`
- Increase swap if system RAM is low

## Customizing Tests

Edit `benchmarks/bench_compare_cpu_gpu.cu` to add custom workloads:

```cpp
// In main() function, add:
bench_accumulate(csv, 50000000, 2048*2048);  // 50M points, 4M cells
bench_pipeline(csv, 25000000, 3072, 768);    // 25M points, custom grid
```

Then rebuild:
```bash
cd build
make bench_compare_cpu_gpu
./bench_compare_cpu_gpu
```

## Sharing Results

To share performance data with your team:

```bash
# Package results
cd build
tar -czf pcr_benchmark_results.tar.gz benchmark_results_*/

# Results include:
# - CSV files (importable to Excel/Tableau)
# - PNG visualizations (ready for presentations)
# - Text logs (for detailed analysis)
```

## Advanced: Profiling with NVIDIA Tools

For detailed GPU analysis:

```bash
# Profile with nvprof (deprecated but simple)
nvprof ./bench_accumulate

# Or use Nsight Compute (modern profiler)
ncu --set full ./bench_accumulate

# Or use Nsight Systems for timeline
nsys profile --stats=true ./bench_tile_pipeline
```

---

**Need help?** See `benchmarks/README.md` for detailed documentation.
