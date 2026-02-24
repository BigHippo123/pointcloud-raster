# PCR — Point Cloud Raster Library

## Project Overview
GPU-accelerated library for rasterizing massive point clouds (500B+ points) into 2D GeoTIFF colormaps. Out-of-core tiled processing with streaming reduction engine.

## Tech Stack
- **Core**: C++17, CUDA (targeting compute capability 7.0+)
- **Build**: CMake 3.18+
- **GPU**: CUB, Thrust (bundled with CUDA toolkit)
- **Geo**: GDAL (GeoTIFF I/O), PROJ (CRS reprojection)
- **Python**: pybind11 bindings, rasterio for Python-side GeoTIFF
- **Test**: GoogleTest (C++), pytest (Python)

## Architecture
Out-of-core tiled streaming reduction engine. Key constraints:
- Output grid can exceed GPU memory (1B+ cells, 4GB+ per band)
- Points stream in chunks, never all in memory
- Tile states checkpoint to disk for crash recovery
- SoA (Struct of Arrays) memory layout throughout for GPU coalescing

### Processing Pipeline
```
PointCloud → CRS Reproject → Filter → TileRouter → per-tile Sort+Reduce → TileManager (disk) → Finalize → GeoTIFF
```

### Reduction Op Contract
Every op provides: `State`, `identity()`, `combine()`, `merge()`, `finalize()`, `state_floats()`.
State is stored as flat floats in band-sequential layout for GPU and disk serialization.

## Key Directories
- `include/pcr/core/` — fundamental types: BBox, CRS, PointCloud (SoA), Grid, GridConfig
- `include/pcr/ops/` — reduction op concept, builtin ops, registry
- `include/pcr/engine/` — GPU kernels and orchestration: accumulator, tile router, filter, tile manager, pipeline
- `include/pcr/io/` — file I/O: GeoTIFF (GDAL), tile state binary format, point cloud formats
- `src/` — implementations mirror include/ structure. `.cu` files for GPU, `.cpp` for CPU
- `python/` — pybind11 bindings + pure Python utilities
- `tests/` — GoogleTest (cpp/) and pytest (python/)
- `benchmarks/` — CUDA perf benchmarks

## Code Style
- C++17 standard, no exceptions in GPU-adjacent code (use Status returns)
- `PCR_HD` macro for `__host__ __device__` functions (empty when `PCR_ENABLE_CUDA=OFF`, defined in reduction_op.h)
- PIMPL pattern for public-facing classes (PointCloud, Grid, Pipeline, etc.)
- Namespace: `pcr::`
- Include guards: `#pragma once`
- Naming: `snake_case` for functions/variables, `PascalCase` for types/classes, `UPPER_CASE` for macros

## Build Commands
```bash
mkdir build && cd build

# CPU-only build (no CUDA required) — use this first
cmake .. -DCMAKE_BUILD_TYPE=Release -DPCR_BUILD_TESTS=ON -DPCR_BUILD_PYTHON=ON -DPCR_ENABLE_CUDA=OFF
make -j$(nproc)

# Full GPU build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPCR_BUILD_TESTS=ON -DPCR_BUILD_PYTHON=ON -DPCR_ENABLE_CUDA=ON
make -j$(nproc)

ctest --output-on-failure    # run C++ tests
cd ../python && pytest       # run Python tests
```

## CPU-Only Build Strategy
When `PCR_ENABLE_CUDA=OFF`:
- `.cu` files are NOT compiled. All GPU kernels are stubbed out.
- Engine files provide CPU fallback implementations (single-threaded loops).
- Use `#ifdef PCR_HAS_CUDA` / `#else` guards in engine headers and source.
- The `PCR_HD` macro becomes empty (no `__host__ __device__`).
- `MemoryLocation::Device` and `MemoryLocation::HostPinned` return errors; only `Host` works.
- `void* stream` parameters are ignored.
- This lets us build and test: types, grid_config, grid, point_cloud, reduction ops (CPU), tile_state_io, grid_io (GDAL), point_cloud_io, and pipeline orchestration logic.
- GPU kernels (accumulator, filter, tile_router, grid_merge, memory_pool) compile as CPU reference implementations first, then get CUDA paths added later.

## Implementation Order (CPU-first)
1. CMakeLists.txt (with PCR_ENABLE_CUDA option, get CPU build working)
2. core/ (types, grid_config, grid, point_cloud) — pure CPU
3. ops/ (builtin_ops, reduction_registry) — CPU reference path
4. io/tile_state_io.cpp — pure CPU file I/O
5. io/grid_io.cpp — GDAL GeoTIFF output (CPU)
6. io/point_cloud_io.cpp — file read/write (CPU)
7. engine/ CPU fallbacks: accumulator, filter, tile_router as single-threaded loops
8. engine/tile_manager.cpp — LRU cache, disk I/O (CPU)
9. engine/pipeline.cpp — orchestration (CPU)
10. python/bindings.cpp — pybind11
11. **--- GPU pass below, add CUDA=ON later ---**
12. engine/memory_pool.cu
13. engine/accumulator.cu — CUB sort+reduce kernel
14. engine/filter.cu + tile_router.cu — GPU kernels
15. engine/grid_merge.cu

## Critical Design Decisions
- Sort-and-reduce (CUB radix sort) over atomics: single code path for all ops
- Template kernels + enum dispatch wrapper for Python interop
- Tile state on disk IS the checkpoint mechanism
- TileManager LRU cache in pinned host memory
- Band-sequential state layout: field f at base[f * num_cells + i]

## Token-Saving Notes for AI
- Headers in include/ are fully written with complete API contracts
- When implementing a .cpp/.cu file, read its corresponding header first
- Don't rewrite headers unless asked — they are the API spec
- Keep implementations focused: one file at a time
- Use /compact between modules to free context
