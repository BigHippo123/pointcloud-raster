# PCR GPU Benchmarking - Docker Quick Start

Run the complete PCR GPU benchmark suite in Docker with a single command!

## Prerequisites

### 1. Docker Installed
```bash
docker --version
# Should show Docker version 20.10 or newer
```

**Install Docker**: https://docs.docker.com/engine/install/

### 2. NVIDIA GPU with Drivers
```bash
nvidia-smi
# Should show your GPU(s) and driver version
```

**Install NVIDIA Drivers**: https://www.nvidia.com/Download/index.aspx

### 3. NVIDIA Container Toolkit
```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
# Should show GPU info from inside container
```

**Install NVIDIA Container Toolkit** (if above fails):

**Ubuntu/Debian:**
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

**Other OS**: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

---

## Quick Start (3 Commands)

```bash
# 1. Build the Docker image (5-10 minutes, one-time)
./docker_run.sh build

# 2. Run the benchmark (1-10 minutes depending on GPU)
./docker_run.sh run

# 3. View results
cd results/performance_plots
open *.png  # or use your image viewer
```

**That's it!** Results are in `./results/`

---

## Detailed Usage

### Build the Image

Build the Docker image with CUDA, dependencies, and PCR library:

```bash
./docker_run.sh build
```

This:
- Uses NVIDIA CUDA 12.2 base image
- Installs CMake, GCC, GDAL, PROJ
- Installs Python with pandas/matplotlib/seaborn
- Builds PCR library with CUDA enabled
- Builds all tests and benchmarks
- Takes ~5-10 minutes (faster on subsequent builds)

### Run Comprehensive Benchmark

Run the full CPU vs GPU comparison:

```bash
./docker_run.sh run
```

This:
- Runs `bench_compare_cpu_gpu` with all test cases
- Generates `performance_results.csv`
- Creates visualizations (6 PNG files)
- Saves everything to `./results/`
- Takes ~1-10 minutes depending on GPU

**Output includes:**
- Console output with real-time results
- `performance_results.csv` - raw data (14 columns)
- `summary_table.csv` - clean comparison table
- `performance_plots/` directory with 6 PNG visualizations
- `benchmark_output.txt` - full console log

### Run Unit Tests

Run all GPU unit tests:

```bash
./docker_run.sh test
```

This runs:
- test_memory_pool (pool allocation, alignment)
- test_grid_merge (init, merge, finalize)
- test_filter (point filtering)
- test_accumulator (all reduction types)
- test_tile_router (tile routing, sorting)

Results saved to `./results/test_results.txt`

### Interactive Shell

Explore the container interactively:

```bash
./docker_run.sh shell
```

Inside the shell:
```bash
# You're in /workspace/build with everything compiled

# Run individual benchmarks
./bench_memory              # Memory pool benchmark
./bench_sort                # CUB sort benchmark
./bench_accumulate          # Accumulate benchmark
./bench_tile_pipeline       # Pipeline benchmark
./bench_compare_cpu_gpu     # Comprehensive comparison

# Run specific tests
./test_memory_pool
./test_grid_merge
./test_accumulator

# Run all tests
ctest --output-on-failure

# Check GPU
nvidia-smi

# Exit when done
exit
```

### Check Prerequisites

Verify your system is ready:

```bash
./docker_run.sh check
```

### Clean Up

Remove the Docker image and build artifacts:

```bash
./docker_run.sh clean
```

---

## Results Structure

After running, `./results/` contains:

```
results/
├── performance_results.csv          # Raw benchmark data
├── summary_table.csv                # Clean comparison table
├── benchmark_output.txt             # Full console output
└── performance_plots/
    ├── wall_time_comparison.png     # CPU vs GPU time
    ├── throughput_comparison.png    # Points/sec comparison
    ├── gpu_speedup.png              # Speedup factors
    ├── resource_usage.png           # RAM and VRAM
    └── utilization.png              # CPU and GPU usage
```

**All PNG files are 300 DPI** (publication quality)

---

## Manual Docker Commands

If you prefer not to use the script:

### Build
```bash
docker build -f Dockerfile.cuda -t pcr-gpu-benchmark:latest .
```

### Run Benchmark
```bash
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    pcr-gpu-benchmark:latest
```

### Interactive Shell
```bash
docker run -it --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -w /workspace/build \
    pcr-gpu-benchmark:latest \
    /bin/bash
```

### Run Tests
```bash
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -w /workspace/build \
    pcr-gpu-benchmark:latest \
    ctest --output-on-failure
```

---

## Customizing the Build

### Change CUDA Architecture

Edit `Dockerfile.cuda` and change:
```dockerfile
ENV CUDA_ARCHITECTURES="70;75;80;86;89"
```

Common values:
- **70**: Tesla V100
- **75**: Tesla T4, RTX 20xx
- **80**: A100
- **86**: RTX 30xx (3060, 3070, 3080, 3090)
- **89**: RTX 40xx (4090), L40

Find your GPU's compute capability: https://developer.nvidia.com/cuda-gpus

### Run Specific Benchmark Only

Create custom run command:
```bash
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -w /workspace/build \
    pcr-gpu-benchmark:latest \
    bash -c "./bench_accumulate > /workspace/results/accumulate_results.txt"
```

### Mount Custom Data

To process your own point cloud data:
```bash
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/data:/data \
    -w /workspace/build \
    pcr-gpu-benchmark:latest \
    /bin/bash
```

---

## Troubleshooting

### "Error: NVIDIA Docker runtime not found"

**Solution**: Install nvidia-container-toolkit (see Prerequisites above)

### "docker: command not found"

**Solution**: Install Docker Desktop or Docker Engine

### "nvidia-smi: command not found"

**Solution**: Install NVIDIA GPU drivers for your card

### "Failed to initialize NVML: Driver/library version mismatch"

**Solution**: Reboot after driver installation, or:
```bash
sudo rmmod nvidia_uvm
sudo rmmod nvidia
sudo modprobe nvidia
```

### Build fails with "No CMAKE_CUDA_COMPILER could be found"

**Solution**: The base image should include CUDA. Try rebuilding:
```bash
./docker_run.sh clean
./docker_run.sh build
```

### Container runs but shows no GPU

**Solution**: Make sure you're using `--gpus all` flag, or update Docker/nvidia-container-toolkit

### Out of memory errors

**Solution**: Reduce test sizes in `benchmarks/bench_compare_cpu_gpu.cu` and rebuild:
```dockerfile
# Edit line numbers in bench_compare_cpu_gpu.cu:
bench_accumulate(csv, 1000000, 1024 * 1024);      # Reduce from 100M
bench_pipeline(csv, 1000000, 2048, 512);          # Reduce from 100M
```

Then:
```bash
./docker_run.sh build
./docker_run.sh run
```

---

## Performance Expectations

### Build Time
- First build: 5-10 minutes (downloads base image, compiles)
- Subsequent builds: 1-2 minutes (uses cache)

### Benchmark Time
- Quick tests (1M points): ~10 seconds
- Full suite (up to 100M points): ~5-10 minutes
- Depends on GPU (V100 faster than T4)

### Expected Speedup
- **Grid Merge**: 20-23x GPU speedup
- **Accumulate**: 22-28x GPU speedup
- **Pipeline**: 20-30x GPU speedup
- See `SAMPLE_EXPECTED_RESULTS.md` for detailed predictions

---

## Docker Image Details

**Base Image**: nvidia/cuda:12.2.0-devel-ubuntu22.04
**Size**: ~8 GB (includes CUDA toolkit, dependencies, build artifacts)
**Contains**:
- Ubuntu 22.04
- CUDA 12.2 toolkit
- CMake 3.22+
- GCC 11
- GDAL 3.4
- PROJ 8.2
- GoogleTest
- Python 3.10 with pandas, matplotlib, seaborn
- PCR library (pre-built)
- All benchmarks (pre-compiled)
- All tests (pre-compiled)

---

## CI/CD Integration

Use in GitHub Actions:

```yaml
name: GPU Benchmark
on: [push]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          ./docker_run.sh build
          ./docker_run.sh run
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: results/
```

---

## Next Steps

After running the benchmark:

1. **Review visualizations**: Open PNG files in `results/performance_plots/`
2. **Analyze CSV**: Import `performance_results.csv` into Excel/Tableau
3. **Check summary**: Open `summary_table.csv` for clean comparison
4. **Share results**: Package `results/` directory for team review
5. **Optimize further**: If speedup is low, check utilization metrics

---

## Questions?

- **Docker issues**: Check Docker/NVIDIA container toolkit docs
- **Build errors**: Share error messages for help
- **Custom benchmarks**: Edit `bench_compare_cpu_gpu.cu` and rebuild
- **More tests**: Add test cases and rebuild

**Documentation**:
- `benchmarks/README.md` - Detailed benchmark guide
- `BENCHMARKING_QUICKSTART.md` - Quick reference
- `SAMPLE_EXPECTED_RESULTS.md` - Performance predictions

---

**Ready to run?**

```bash
./docker_run.sh build
./docker_run.sh run
```

Results will appear in `./results/` when complete! 🚀
