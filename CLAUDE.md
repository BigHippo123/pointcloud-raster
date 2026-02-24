# PCR — Point Cloud Raster Library (AI Agent Quick Reference)

GPU-accelerated library for rasterizing massive point clouds into 2D GeoTIFF colormaps.

## Quick Build Commands

```bash
# Full build (CUDA required)
cmake --build /workspace/build --target all -j$(nproc)

# Python extension only (faster iteration)
cmake --build /workspace/build --target _pcr -j$(nproc)

# CPU-only build (no GPU)
cd /workspace/build
cmake .. -DPCR_ENABLE_CUDA=OFF
make -j$(nproc)
```

## Test Commands

```bash
# C++ tests
cd /workspace/build && ctest --output-on-failure

# Python tests
cd /workspace/python && PYTHONPATH=/workspace/python pytest

# Quick import check
PYTHONPATH=/workspace/python python3 -c "import pcr; print(pcr.__version__)"
```

## Key Directories

| Directory | Purpose |
|-----------|---------|
| `include/pcr/` | Public headers (core/, ops/, engine/, io/) |
| `src/` | Implementation (.cpp, .cu files mirror include/ structure) |
| `python/` | Python bindings (bindings.cpp) and package |
| `tests/cpp/` | GoogleTest C++ tests |
| `tests/python/` | pytest Python tests |
| `scripts/` | Organized into benchmarks/, patterns/, diagnostics/, data/, shell/ |
| `docs/` | All documentation (see docs/README.md for index) |
| `benchmarks/` | CUDA performance benchmarks |
| `examples/python/` | Python usage examples |

## Code Conventions

- **C++ Standard**: C++17
- **Naming**: `snake_case` for functions/variables, `PascalCase` for types
- **Memory layout**: SoA (Struct of Arrays) throughout
- **GPU functions**: Use `PCR_HD` macro for `__host__ __device__`
- **Error handling**: Return `Status` objects (no exceptions in GPU code)
- **Includes**: `#pragma once` for guards

## Key Files

| File | Purpose |
|------|---------|
| `include/pcr/engine/pipeline.h` | ExecutionMode enum, PipelineConfig struct |
| `src/engine/pipeline.cpp` | Pipeline implementation (CPU, GPU, Hybrid modes) |
| `python/bindings.cpp` | All Python-exposed classes |
| `CMakeLists.txt` | Build configuration (PCR_ENABLE_CUDA, PCR_BUILD_PYTHON, etc.) |

## Execution Modes

- **CPU**: OpenMP multithreading (controlled by `cpu_threads`)
- **GPU**: CUDA kernels (requires `PCR_ENABLE_CUDA=ON`)
- **Auto**: GPU if available, else CPU
- **Hybrid**: CPU routing + GPU accumulation (producer-consumer pattern)

## Full Documentation

See [docs/CLAUDE.md](docs/CLAUDE.md) for complete architecture guide, implementation details, and design decisions.
