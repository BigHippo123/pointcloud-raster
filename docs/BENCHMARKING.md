# PCR Benchmarking Guide

Complete guide to running performance benchmarks, generating reports, and interpreting results.

## Quick Start

```bash
# Build the library first
cmake --build /workspace/build --target _pcr -j$(nproc)

# Run all benchmarks and generate HTML + Markdown report (~5–10 min on CPU, ~2 min with GPU)
scripts/shell/run_benchmarks.sh

# Skip the slow full-glyph benchmark (~60 s, uses quick chart only)
scripts/shell/run_benchmarks.sh --quick

# Force CPU-only (no GPU even if available)
scripts/shell/run_benchmarks.sh --no-gpu
```

Results are written to `benchmark_results/<TIMESTAMP>/` and a `benchmark_results/latest/` symlink is updated.

---

## What Gets Measured

The suite runs six benchmark scripts in order:

| Script | Output | Measures |
|--------|--------|---------|
| `benchmark_multithread.py` | `multithread.csv` | CPU thread-count scaling (1T → N-T vs GPU) |
| `benchmark_cpu_gpu.py` | `cpu_vs_gpu.csv` | CPU vs GPU throughput at 1M / 5M / 10M / 25M points |
| `benchmark_hybrid.py` | `hybrid.csv` | Hybrid mode (CPU routing + GPU accumulation) vs baselines |
| `bench_glyphs.py` | `bench_glyphs_chart.png` | Quick glyph sweep (~60 s) |
| `benchmark_glyph_full.py` | `glyphs_full.csv` + chart | Full glyph × footprint × N sweep (skipped with `--quick`) |
| `generate_glyph_patterns.py` | `glyph_01_*.png` … `glyph_08_*.png` | Visual gallery of glyph types |

An optional DC LiDAR step runs if `$DC_LAS_DIR` points to a directory of LAS files:

```bash
DC_LAS_DIR=/data/dc-lidar scripts/shell/run_benchmarks.sh
```

---

## Report Generation

After the benchmarks finish, `generate_report.py` assembles all CSVs and PNGs into two report files:

```
benchmark_results/<TIMESTAMP>/
├── benchmark_report.html   ← standalone HTML with all images embedded as base64
└── benchmark_report.md     ← GitHub-flavored Markdown (images as relative paths)
```

You can regenerate the report at any time from an existing results directory:

```bash
PYTHONPATH=/workspace/python python3 scripts/benchmarks/generate_report.py \
  --results-dir benchmark_results/latest \
  --title "My Machine — RTX 4090"
```

---

## Individual Benchmarks

### CPU vs GPU Throughput

```bash
PYTHONPATH=/workspace/python python3 scripts/benchmarks/benchmark_cpu_gpu.py
```

Measures wall-clock time for the full ingest + finalize cycle at four point counts (1M / 5M / 10M / 25M). Outputs a CSV with columns `points, mode, elapsed_s, mpts_per_s, speedup`.

### Multi-Thread Scaling

```bash
PYTHONPATH=/workspace/python python3 scripts/benchmarks/benchmark_multithread.py
```

Sweeps CPU thread count (1, 2, 4, N) and adds a GPU row if available. Shows whether additional cores help at the given memory bandwidth.

### Hybrid Mode

```bash
PYTHONPATH=/workspace/python python3 scripts/benchmarks/benchmark_hybrid.py
```

Compares the producer-consumer Hybrid mode against CPU-MT and GPU baselines. Hybrid is most useful when GPU memory is limited and the tile state doesn't fit entirely on-device.

### Glyph Throughput

```bash
# Quick (all glyph types, small grid, ~60 s)
PYTHONPATH=/workspace/python python3 scripts/benchmarks/bench_glyphs.py

# Full (1000×1000 grid, 100K/1M/5M pts, best-of-3, ~5–20 min)
PYTHONPATH=/workspace/python python3 scripts/benchmarks/benchmark_glyph_full.py \
  --mode cpu --sizes 100000 1000000 5000000 --repeats 3 --timeout 30
```

### Visual Glyph Patterns

```bash
PYTHONPATH=/workspace/python python3 scripts/patterns/generate_glyph_patterns.py --mode cpu
```

Generates eight PNG images in `glyph_pattern_outputs/` illustrating how each glyph type behaves. See [Glyph Visual Gallery](#glyph-visual-gallery) below.

---

## Glyph Types: Performance Guide

All three glyph types support the same reduction operations (`Average`, `WeightedAverage`, `Sum`, `Max`, `Min`, `Count`). They differ in footprint size and therefore cost.

### Point (baseline)

Each point writes to exactly one cell. Zero footprint overhead. Use as the baseline for comparison.

- **Cost**: O(N) regardless of grid density
- **Best for**: High-density clouds where every cell gets multiple hits
- **CPU**: ~2–4 Mpts/s on a modern 6-core machine
- **GPU**: 25–110 Mpts/s (scales almost linearly with point count)

### Line (`line_splat_spec`)

Each point paints a Bresenham segment of length `2 × half_length` cells, oriented by a per-point `direction` channel.

```python
spec = pcr.line_splat_spec(
    'value',
    direction_channel='direction',       # radians, per-point
    half_length_channel='half_length',   # cells, per-point (or use default_half_length)
    default_half_length=5.0,
    max_radius_cells=7.0,
)
```

- **Cost**: O(N × half_length). `hl=1` ≈ Point speed; `hl=16` is ~2× slower
- **Best for**: Elongated features — roads, scan lines, flow vectors
- **Tip**: keep `max_radius_cells ≈ half_length + 2` to avoid wasted kernel iterations

### Gaussian (`gaussian_splat_spec`)

Each point applies a 2D Gaussian kernel of radius `max_radius_cells`, with per-point σ and optional anisotropy (σx ≠ σy) and rotation.

```python
spec = pcr.gaussian_splat_spec(
    'value',
    default_sigma=3.0,          # cells; use sigma_x_channel / sigma_y_channel for per-point
    max_radius_cells=12.0,      # ≈ 4 × sigma is a good rule of thumb
)
```

- **Cost**: O(N × σ²). σ=1: fast; σ=4: ~20× slower; σ=16: ~900× slower on CPU
- **Best for**: Gap-filling sparse LiDAR tiles; smooth interpolation
- **GPU advantage is largest here** — atomic contention at σ=4+ on CPU serialises threads; GPU absorbs it
- **Tip**: for σ > 8 on CPU, use `--timeout 30` in the benchmark to skip unreasonably slow cells

#### Footprint cost summary (CPU, 1M points)

| Glyph | Config | Time (s) | Mpts/s |
|-------|--------|----------|--------|
| Point | — | ~0.33 | ~3.0 |
| Line | hl=1 | ~0.30 | ~3.4 |
| Line | hl=4 | ~0.31 | ~3.3 |
| Line | hl=16 | ~0.44 | ~2.3 |
| Gaussian | σ=1 | ~0.75 | ~1.3 |
| Gaussian | σ=4 | ~6.4 | ~0.16 |
| Gaussian | σ=16 | >30 s | <0.05 |

*Measured on i5-9400F (6-core, 2.9 GHz), 1000×1000 grid. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for full tables.*

---

## Glyph Visual Gallery

The gallery script (`scripts/patterns/generate_glyph_patterns.py`) produces eight PNG files:

| File | What it shows |
|------|---------------|
| `01_gap_fill_comparison.png` | Point (blocky, gaps) vs Gaussian σ=2 vs σ=6 on a sparse cloud |
| `02_sigma_progression.png` | Gaussian σ from 0.5 to 16 on a sinusoidal surface |
| `03_anisotropic_gaussian.png` | Isotropic, X/Y-elongated, and rotated (30°/45°/75°) kernels |
| `04_line_directions.png` | 9-direction sweep + half-length 2 → 32 progression |
| `05_flow_field.png` | Vortex flow field — Line glyph vs Adaptive Gaussian |
| `06_sparse_vs_dense.png` | N=50 / 500 / 5,000 points — Point vs Gaussian σ=2.5 |
| `07_per_point_sigma.png` | Per-point σ proportional to distance from cloud centre |
| `08_glyph_showcase.png` | Bullseye concentric rings — Point / Line / Gauss σ=2 / Gauss σ=5 |

---

## DC LiDAR Real-World Test

The optional DC LiDAR step runs `scripts/data/test_dc_lidar.py` against real LAS files from Washington DC. It streams tiles one at a time and measures:

- **I/O time** (laspy file read)
- **Library ingest time** (PCR pipeline)
- **Finalize time** (tile merge + GeoTIFF write)

It also runs each glyph variant (`point`, `line-2`, `line-5`, `gaussian-1`, `gaussian-3`, `gaussian-5`, `gaussian-10`) and saves per-glyph TIF files which are then assembled into a visual comparison PNG by `render_dc_tifs.py`.

```bash
# With a DC LiDAR directory
DC_LAS_DIR=/data/dc-lidar scripts/shell/run_benchmarks.sh

# Or run directly
PYTHONPATH=/workspace/python python3 scripts/data/test_dc_lidar.py \
  --las-dir /data/dc-lidar --mode cpu-mt --glyph all --subset 5

# Render the comparison PNG separately
PYTHONPATH=/workspace/python python3 scripts/benchmarks/render_dc_tifs.py \
  --tif-dir glyph_pattern_outputs --mode cpu-mt \
  --output benchmark_results/latest/dc_comparison.png
```

See [DC_LIDAR_BENCHMARK_README.md](DC_LIDAR_BENCHMARK_README.md) for dataset details and the full 188-file results.

---

## Output File Reference

```
benchmark_results/<TIMESTAMP>/
├── system_info.txt          key=value system metadata
├── multithread.csv          thread-scaling results
├── cpu_vs_gpu.csv           CPU vs GPU throughput
├── hybrid.csv               Hybrid mode comparison
├── glyphs_full.csv          glyph × footprint × N throughput
├── glyphs_full_chart.png    bar chart of glyph throughput
├── bench_glyphs_chart.png   quick glyph chart
├── glyph_01_*.png           visual gallery (8 images)
│   …
├── glyph_08_*.png
├── dc_lidar.csv             real-data mode comparison (optional)
├── dc_comparison.png        glyph comparison rendered from TIFs (optional)
├── benchmark_report.md      GitHub-flavored Markdown report
└── benchmark_report.html    standalone HTML (all images embedded)

benchmark_results/latest/    symlink → most recent run
```

---

## Troubleshooting

**`Cannot import pcr`** — rebuild the Python extension:
```bash
cmake --build /workspace/build --target _pcr -j$(nproc)
```

**GPU not detected** — check `nvidia-smi` and CUDA library paths. The suite gracefully falls back to CPU-only if no GPU is found.

**`Gauss σ=16` takes forever** — expected on CPU. The default `--timeout 30` in the full benchmark skips cells that exceed 30 s on warmup. Lower sigma or use GPU.

**Missing `matplotlib`** — charts are silently skipped. Install with `pip install matplotlib`.

**Missing `laspy`** — the DC LiDAR step requires `pip install laspy lazrs-python` for LAZ support; uncompressed LAS files work without it.
