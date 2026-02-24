# PCR — Point Cloud Reduction

A high-performance CUDA/C++ library for reducing large point clouds onto regular grids, with Python bindings and support for GPU acceleration, CPU multi-threading, and glyph splatting.

---

## Installation

### From PyPI (Recommended)

```bash
pip install pcr
```

**Note:** Pre-built wheels are coming soon. For now, install from source.

### From Source

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed build instructions.

Quick build:
```bash
git clone https://github.com/pcr-dev/pcr.git
cd pcr
pip install -r requirements-dev.txt
mkdir build && cd build
cmake .. -DPCR_ENABLE_CUDA=ON -DBUILD_PYTHON=ON
cmake --build . -j$(nproc)
cd ../python && pip install -e .
```

---

## Features

- **GPU-accelerated** rasterization via CUDA (10–200× faster than single-threaded CPU for large datasets)
- **CPU multi-threading** via OpenMP (scales linearly with core count)
- **Glyph splatting** — each point can paint a weighted footprint across multiple cells:
  - **Point** — single-cell scatter (baseline, zero overhead)
  - **Line** — Bresenham segment, direction + half-length per point
  - **Gaussian** — smooth kernel with per-point σ and rotation
- **Multiple execution modes**: `CPU`, `GPU`, `Auto`, `Hybrid`
- **Pluggable reduction ops**: `Sum`, `Average`, `WeightedAverage`, `Max`, `Min`, `Count`
- **Per-point glyph parameters** stored as named channels (`sigma`, `direction`, `half_length`, …)
- **Python bindings** via PyBind11; numpy arrays in/out
- GeoTIFF output via GDAL

---

## Build

### Requirements

- CMake ≥ 3.18
- C++17 compiler
- CUDA toolkit (optional, enables GPU)
- OpenMP (optional, enables CPU threading — usually bundled with GCC/Clang)
- GDAL
- PyBind11 + Python 3.8+

### Build commands

```bash
mkdir build && cd build

# With GPU support
cmake -DPCR_ENABLE_CUDA=ON ..

# CPU only
cmake ..

cmake --build . -j$(nproc)

# Rebuild only the Python extension
cmake --build . --target _pcr -j$(nproc)
```

The Python extension is written to `python/pcr/_pcr.cpython-310-x86_64-linux-gnu.so`.

### Run C++ tests

```bash
ctest --test-dir build -j$(nproc)
```

All 17 test suites pass.

---

## Quick Start (Python)

```python
import pcr
import numpy as np

# --- Grid definition ---
bbox = pcr.BBox()
bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y = 0, 0, 1000, 1000
gc = pcr.GridConfig()
gc.bounds = bbox
gc.cell_size_x =  1.0
gc.cell_size_y = -1.0   # north-up raster
gc.compute_dimensions()

# --- Point cloud ---
N = 1_000_000
rng = np.random.default_rng(42)
cloud = pcr.PointCloud.create(N)
cloud.set_x_array(rng.uniform(0, 1000, N))
cloud.set_y_array(rng.uniform(0, 1000, N))
cloud.add_channel('value', pcr.DataType.Float32)
cloud.set_channel_array_f32('value', rng.uniform(0, 1, N).astype(np.float32))

# --- Reduction spec (Point glyph — fastest) ---
spec = pcr.ReductionSpec()
spec.value_channel = 'value'
spec.type = pcr.ReductionType.Average
spec.output_band_name = 'mean_value'

# --- Pipeline ---
cfg = pcr.PipelineConfig()
cfg.grid = gc
cfg.reductions = [spec]
cfg.exec_mode = pcr.ExecutionMode.GPU   # or CPU / Auto / Hybrid
cfg.gpu_fallback_to_cpu = True
cfg.state_dir = '/tmp/pcr_state'
cfg.output_path = '/tmp/output.tif'

pipe = pcr.Pipeline.create(cfg)
pipe.ingest(cloud)
pipe.finalize()
```

### Gaussian glyph (smooth gap-filling)

```python
spec = pcr.gaussian_splat_spec(
    'value',
    default_sigma=3.0,        # cells
    max_radius_cells=12.0,
)
```

### Line glyph (elongated footprint)

```python
cloud.add_channel('direction',   pcr.DataType.Float32)
cloud.add_channel('half_length', pcr.DataType.Float32)
cloud.set_channel_array_f32('direction',   directions_radians)
cloud.set_channel_array_f32('half_length', np.full(N, 5.0, np.float32))

spec = pcr.line_splat_spec(
    'value',
    direction_channel='direction',
    half_length_channel='half_length',
    max_radius_cells=8.0,
)
```

### Per-point sigma (heterogeneous footprints)

```python
cloud.add_channel('sigma', pcr.DataType.Float32)
cloud.set_channel_array_f32('sigma', sigma_per_point)

spec = pcr.gaussian_splat_spec(
    'value',
    sigma_x_channel='sigma',
    sigma_y_channel='sigma',
    max_radius_cells=32.0,
)
```

---

## Execution Modes

| Mode | Description |
|------|-------------|
| `CPU` | OpenMP threads, all cores by default (`cpu_threads=0`) |
| `GPU` | CUDA kernels; falls back to CPU if no device found |
| `Auto` | GPU if available, otherwise CPU |
| `Hybrid` | CPU routing threads + GPU accumulation (producer-consumer) |

---

## Performance

Measured on a **1000×1000 grid (1M cells)**, best-of-3 timed runs (`scripts/benchmarks/benchmark_glyph_full.py`).

### Throughput (Mpts/s)

| Glyph | N=100k CPU | N=100k GPU | N=1M CPU | N=1M GPU | N=5M CPU | N=5M GPU |
|-------|-----------|-----------|---------|---------|---------|---------|
| Point | 2.60 | 3.44 | 2.94 | 27.11 | 1.72 | **60.26** |
| Line hl=1 | 2.42 | 3.91 | 3.39 | 25.57 | 1.74 | **50.60** |
| Line hl=4 | 2.54 | 3.86 | 3.11 | 23.24 | 1.64 | **50.41** |
| Line hl=16 | 2.13 | 3.46 | 2.30 | 22.93 | 1.37 | **46.61** |
| Gauss σ=1 | 1.20 | 3.42 | 1.30 | 22.79 | 0.94 | **47.40** |
| Gauss σ=4 | 0.16 | 3.10 | 0.15 | 14.79 | 0.15 | **21.59** |
| Gauss σ=16 | 0.01 | 1.04 | 0.01 | 2.68 | 0.01 | **2.43** |

### GPU Speedup at N=5M

| Glyph | CPU time | GPU time | Speedup |
|-------|---------|---------|---------|
| Point | 2.9 s | 0.08 s | **35×** |
| Line hl=16 | 3.7 s | 0.11 s | **34×** |
| Gauss σ=1 | 5.3 s | 0.11 s | **51×** |
| Gauss σ=4 | 33.7 s | 0.23 s | **145×** |
| Gauss σ=16 | **442 s** | 2.1 s | **215×** |

GPU advantage grows with footprint size because atomic contention scales with σ² — the GPU's massive parallelism absorbs it far better than CPU. Gauss σ=16 CPU at 5M points takes 7+ minutes; GPU takes 2 seconds.

### Run the benchmarks

```bash
# Full automated suite — generates HTML + Markdown report in benchmark_results/
scripts/shell/run_benchmarks.sh

# Quick run (skips the slow full-glyph sweep, ~60 s total)
scripts/shell/run_benchmarks.sh --quick

# Or run individual scripts
PYTHONPATH=python python3 scripts/benchmarks/bench_glyphs.py
PYTHONPATH=python python3 scripts/benchmarks/benchmark_glyph_full.py
```

See [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for measured numbers and [docs/BENCHMARKING.md](docs/BENCHMARKING.md) for the full guide.

For benchmark review with example pictures review [git pages](https://bighippo123.github.io/pointcloud-raster/)

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/shell/run_benchmarks.sh` | **Full benchmark suite** — runs all scripts, generates HTML + Markdown report |
| `scripts/benchmarks/bench_glyphs.py` | Quick glyph benchmark (~60–90 s); produces bar chart |
| `scripts/benchmarks/benchmark_glyph_full.py` | Comprehensive benchmark: ETA, timeout guard, CSV + chart |
| `scripts/benchmarks/benchmark_cpu_gpu.py` | CPU vs GPU throughput comparison |
| `scripts/benchmarks/benchmark_multithread.py` | CPU thread-count scaling |
| `scripts/benchmarks/benchmark_hybrid.py` | Hybrid vs CPU-MT vs GPU |
| `scripts/benchmarks/generate_report.py` | Assembles CSVs + PNGs into `benchmark_report.html` / `.md` |
| `scripts/patterns/generate_glyph_patterns.py` | Generates 8 visual pattern PNGs showing glyph effects |
| `scripts/data/test_dc_lidar.py` | Real-data test: streams DC LiDAR LAS files, all glyph types |

---

## Real-Data Test — Washington DC LiDAR

`scripts/data/test_dc_lidar.py` validates the pipeline on a real aerial LiDAR dataset
(Washington DC, NAD83/Maryland State Plane, EPSG:32110).

### Dataset

- **188 LAS 1.4 tiles**, ~800 m × 800 m each, ~14 GB total
- **479,541,463 points** (479M), point format 6 (elevation + intensity + classification)
- Coverage: 10 km × 24 km at ~4–5 pts/m²

### Full-dataset results (GPU, σ=3 cells, 1 m cell size)

| | Point glyph | Gaussian σ=3 |
|---|---|---|
| Grid | 10,402 × 24,002 | 10,402 × 24,002 |
| Grid cells | 249.7M | 249.7M |
| Ingest throughput | 18.1 Mpts/s | 17.6 Mpts/s |
| Ingest time | 26.5 s | 27.3 s |
| Finalize time | 23.8 s | 23.4 s |
| Disk read time | 165 s | 166 s |
| **Coverage** | **41.8%** | **45.8%** |
| Elevation range (5–95th pct) | 1.7 – 110.9 m | −0.1 – 117.6 m |
| Total wall time | ~228 s | ~227 s |

Disk I/O dominates (165 s read vs 27 s GPU ingest) — the pipeline is fast enough
that reading 14 GB from disk is the bottleneck. Gaussian σ=3 recovers an extra 4%
coverage by filling gaps between sparse tiles.

### Usage

```bash
# Full dataset (all 188 files, ~7.5 min)
PYTHONPATH=python python3 scripts/data/test_dc_lidar.py --mode gpu

# Quick single-tile test (~5 s)
PYTHONPATH=python python3 scripts/data/test_dc_lidar.py --subset 1

# Ground points only, custom sigma
PYTHONPATH=python python3 scripts/data/test_dc_lidar.py --ground-only --sigma 5

# Intensity instead of elevation
PYTHONPATH=python python3 scripts/data/test_dc_lidar.py --value intensity
```

Outputs: `dc_lidar_point.tif`, `dc_lidar_gaussian.tif`,
`dc_lidar_comparison.png`, `dc_lidar_stats.json`.

The LAS reader uses pure numpy (no `laspy` required) — files are uncompressed
LAS 1.4 format 6 read directly with a structured dtype. For LAZ support,
install `laspy` and `lazrs-python`.

---

## Glyph Visual Patterns

Run to generate PNG images in `glyph_pattern_outputs/`:

```bash
PYTHONPATH=python python3 scripts/patterns/generate_glyph_patterns.py --mode cpu
```

Images produced:

| File | Shows |
|------|-------|
| `01_gap_fill_comparison.png` | Point (blocky, 8% coverage) vs Gaussian σ=2 vs σ=6 on sparse data |
| `02_sigma_progression.png` | Gaussian σ from 0.5 to 16 on a sinusoidal surface |
| `03_anisotropic_gaussian.png` | Isotropic, X/Y elongated, 30°/45°/75° rotated kernels |
| `04_line_directions.png` | 9-direction sweep + half-length scaling |
| `05_flow_field.png` | Vortex flow with line glyphs + adaptive Gaussian density |
| `06_sparse_vs_dense.png` | N=50/500/5000 points × Point vs Gaussian |
| `07_per_point_sigma.png` | σ proportional to distance from center |
| `08_glyph_showcase.png` | Bullseye rings — Point / Line / Gauss σ=2 / Gauss σ=5 |

---

## Key Files

| Path | Purpose |
|------|---------|
| `include/pcr/engine/pipeline.h` | `ExecutionMode`, `PipelineConfig`, `ReductionSpec` |
| `include/pcr/engine/glyph.h` | `GlyphType`, `GlyphSpec` |
| `include/pcr/engine/glyph_kernels.h` | CPU + GPU glyph accumulator declarations |
| `src/engine/glyph_kernels.cu` | Gaussian + Line CUDA kernels and CPU reference |
| `src/engine/pipeline.cpp` | Pipeline::Impl — ingest/finalize orchestration |
| `src/engine/tile_router.cpp` | Point sorting, tile dispatch, glyph co-sort |
| `src/ops/reduction_registry.cpp` | Reduction op registry (Sum/Avg/WeightedAvg/…) |
| `python/bindings.cpp` | PyBind11 bindings |
| `python/pcr/__init__.py` | Python helpers: `gaussian_splat_spec`, `line_splat_spec` |

---

## Known Limitations / Future Work

- Large Gaussian footprints (σ > 16 cells) are slow on CPU due to atomic contention — use GPU
- Line glyphs are 1-pixel wide (Bresenham); anti-aliased lines are future work
- Arbitrary user-defined weight functions (Phase 10) are not yet implemented
- Stress-tested up to 479M points on real DC LiDAR data; streaming billions requires chunked ingest integration
- LAZ (compressed) LAS support requires `pip install laspy lazrs-python`
- Container deployment needs CUDA device passthrough configuration

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Before contributing:
- Run tests: `ctest --test-dir build && pytest tests/python`
- Format code: `black python/ tests/python/` (Python) or `clang-format` (C++)
- Follow conventional commits: `feat:`, `fix:`, `docs:`, etc.
