#!/usr/bin/env python3
"""
Comprehensive glyph benchmark: Point / Line / Gaussian  ×  CPU / GPU  ×  N  ×  footprint-size.

Measures wall-clock time for ingest + finalize (pipeline already created before
timing starts so one-time init cost is excluded).

Dimensions swept:
  - Glyph type:   Point (baseline), Line, Gaussian
  - Footprint:    Line half_length  ∈ {1,4,16}  /  Gaussian sigma  ∈ {1,4,16}
  - Points N:     configurable, default 100k / 1M / 5M
  - Mode:         CPU (all-core OpenMP) and GPU

Each (glyph, footprint, N, mode) cell is repeated REPEATS times; best-of is
reported (eliminates OS jitter). A 1-run warmup is performed first and excluded.

Output:
  - Formatted table to stdout
  - CSV to  scripts/benchmark_glyph_results.csv
  - Bar chart PNG to  scripts/benchmark_glyph_chart.png

Usage:
    python3 scripts/benchmark_glyph_full.py
    python3 scripts/benchmark_glyph_full.py --sizes 100000 1000000 --repeats 5
    python3 scripts/benchmark_glyph_full.py --mode cpu   # skip GPU
"""

import os
import sys
import csv
import time
import shutil
import argparse
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import pcr

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bbox(w, h):
    b = pcr.BBox()
    b.min_x, b.min_y, b.max_x, b.max_y = 0.0, 0.0, float(w), float(h)
    return b


def make_gc(bbox, cell_size=1.0):
    gc = pcr.GridConfig()
    gc.bounds = bbox
    gc.cell_size_x = cell_size
    gc.cell_size_y = -cell_size
    gc.compute_dimensions()
    return gc


def make_cloud(n, bbox, rng):
    """Random cloud with value + sigma + direction + half_length channels."""
    xs = rng.uniform(bbox.min_x + 2, bbox.max_x - 2, n)
    ys = rng.uniform(bbox.min_y + 2, bbox.max_y - 2, n)
    cloud = pcr.PointCloud.create(n)
    cloud.set_x_array(xs)
    cloud.set_y_array(ys)
    cloud.add_channel('value',       pcr.DataType.Float32)
    cloud.add_channel('sigma',       pcr.DataType.Float32)
    cloud.add_channel('direction',   pcr.DataType.Float32)
    cloud.add_channel('half_length', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value',       rng.uniform(0, 1, n).astype(np.float32))
    cloud.set_channel_array_f32('sigma',       np.full(n, 2.0, np.float32))
    cloud.set_channel_array_f32('direction',   rng.uniform(0, np.pi, n).astype(np.float32))
    cloud.set_channel_array_f32('half_length', np.full(n, 2.0, np.float32))
    return cloud


def build_pipeline(spec, gc, mode, state_dir, out_path):
    cfg = pcr.PipelineConfig()
    cfg.grid = gc
    cfg.reductions = [spec]
    cfg.exec_mode = mode
    cfg.gpu_fallback_to_cpu = True
    cfg.state_dir = state_dir
    cfg.output_path = out_path
    shutil.rmtree(state_dir, ignore_errors=True)
    return pcr.Pipeline.create(cfg)


def time_run(pipe, cloud):
    """Ingest + finalize one pass. State dir is managed by the pipeline."""
    t0 = time.perf_counter()
    pipe.ingest(cloud)
    pipe.finalize()
    return time.perf_counter() - t0


def eta_str(elapsed, done, total):
    if done == 0:
        return '?'
    remaining = elapsed / done * (total - done)
    m, s = divmod(int(remaining), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


# ---------------------------------------------------------------------------
# Spec factories
# ---------------------------------------------------------------------------

def spec_point():
    s = pcr.ReductionSpec()
    s.value_channel = 'value'
    s.type = pcr.ReductionType.Average
    s.output_band_name = 'out'
    return s

def spec_line(hl):
    return pcr.line_splat_spec(
        'value',
        direction_channel='direction',
        half_length_channel='half_length' if hl == 'channel' else '',
        default_half_length=float(hl) if hl != 'channel' else 2.0,
        max_radius_cells=float(hl) + 2 if hl != 'channel' else 4.0,
    )

def spec_gaussian(sigma):
    return pcr.gaussian_splat_spec(
        'value',
        default_sigma=float(sigma),
        max_radius_cells=min(4.0 * sigma, 64.0),
    )


# ---------------------------------------------------------------------------
# Benchmark suite definition
# ---------------------------------------------------------------------------

# Each entry: (label, factory_fn, footprint_param, footprint_label)
# footprint_param is passed to the factory; None = no footprint sweep.
SUITE = [
    # Point baseline
    ('Point',            spec_point,    None,  'N/A'),
    # Line: three half-lengths
    ('Line  hl=1',       spec_line,     1,     'hl=1 '),
    ('Line  hl=4',       spec_line,     4,     'hl=4 '),
    ('Line  hl=16',      spec_line,     16,    'hl=16'),
    # Gaussian: three sigmas
    ('Gauss σ=1',        spec_gaussian, 1,     'σ=1  '),
    ('Gauss σ=4',        spec_gaussian, 4,     'σ=4  '),
    ('Gauss σ=16',       spec_gaussian, 16,    'σ=16 '),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes',   nargs='+', type=int,
                        default=[100_000, 1_000_000, 5_000_000])
    parser.add_argument('--mode',    choices=['cpu', 'gpu', 'both'], default='both')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Timed repetitions per cell (best-of is reported)')
    parser.add_argument('--grid',    type=int, default=1000,
                        help='Grid side length in cells (NxN)')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='Skip a cell if warmup takes longer than this many seconds')
    args = parser.parse_args()

    modes = []
    if args.mode in ('cpu', 'both'):
        modes.append(('CPU', pcr.ExecutionMode.CPU))
    if args.mode in ('gpu', 'both'):
        modes.append(('GPU', pcr.ExecutionMode.GPU))

    rng   = np.random.default_rng(42)
    bbox  = make_bbox(args.grid, args.grid)
    gc    = make_gc(bbox)
    tmpdir = tempfile.mkdtemp(prefix='pcr_bench_glyph_')

    print(f"\nGrid         : {gc.width} × {gc.height} = {gc.width*gc.height:,} cells")
    print(f"Sizes        : {[f'{n:,}' for n in args.sizes]}")
    print(f"Modes        : {[m for m, _ in modes]}")
    print(f"Repeats      : {args.repeats} (best-of reported)")

    # Pre-build clouds for all sizes
    print("\nBuilding point clouds ...", flush=True)
    clouds = {n: make_cloud(n, bbox, rng) for n in args.sizes}

    mode_labels = [m for m, _ in modes]
    col_w = 10
    total_cells = len(args.sizes) * len(SUITE) * len(modes)

    sep = '─' * (22 + 12 + col_w * len(modes) * 2 + 4 * len(modes))
    hdr = f"\n{'Glyph':<22} {'N':>12}" + ''.join(
              f"  {ml+' t(s)':>{col_w}}  {ml+' Mpt/s':>{col_w}}"
              for ml in mode_labels)
    print('\n' + sep)
    print(hdr)
    print(sep)

    results = []
    done_cells = 0
    bench_start = time.perf_counter()

    try:
        for n in args.sizes:
            cloud = clouds[n]
            state_dir = os.path.join(tmpdir, 'state')
            out_path  = os.path.join(tmpdir, 'out.tif')
            row_times = {}

            for label, factory, param, _ in SUITE:
                spec = factory(param) if param is not None else factory()

                for mode_name, exec_mode in modes:
                    elapsed_total = time.perf_counter() - bench_start
                    eta = eta_str(elapsed_total, done_cells, total_cells)
                    print(f"  [{done_cells+1}/{total_cells}] {label} / {mode_name} / n={n:,}  ETA {eta} ...",
                          end='\r', flush=True)

                    pipe = build_pipeline(spec, gc, exec_mode, state_dir, out_path)
                    if pipe is None:
                        row_times[(label, mode_name)] = None
                        done_cells += 1
                        continue

                    # Warmup — if it exceeds timeout, mark as SLOW and skip repeats
                    t_warmup = time_run(pipe, cloud)
                    done_cells += 1

                    if t_warmup > args.timeout:
                        print(f"  [{done_cells}/{total_cells}] {label} / {mode_name} / n={n:,}  "
                              f"SKIP (warmup {t_warmup:.1f}s > timeout {args.timeout:.0f}s)")
                        row_times[(label, mode_name)] = None
                        continue

                    best = t_warmup
                    for _ in range(args.repeats - 1):
                        best = min(best, time_run(pipe, cloud))

                    row_times[(label, mode_name)] = best
                    results.append((label, n, mode_name, best, n / best / 1e6))

                # Print completed row (clear progress line first)
                print(' ' * 80, end='\r')
                cells = ''
                for mode_name, _ in modes:
                    t = row_times.get((label, mode_name))
                    if t is None:
                        cells += f"  {'SLOW':>{col_w}}  {'---':>{col_w}}"
                    else:
                        cells += f"  {t:>{col_w}.3f}  {n/t/1e6:>{col_w}.2f}"
                print(f"{label:<22} {n:>12,}{cells}")

            print(sep)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    total_elapsed = time.perf_counter() - bench_start
    m, s = divmod(int(total_elapsed), 60)
    print(f"\nTotal time: {m}m{s:02d}s")

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    csv_path = os.path.join(HERE, 'benchmark_glyph_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['glyph', 'n_points', 'mode', 'elapsed_s', 'mpts_per_s'])
        w.writerows(results)
    print(f"\nCSV: {csv_path}")

    # ------------------------------------------------------------------
    # Chart
    # ------------------------------------------------------------------
    _plot(results, args.sizes, mode_labels,
          os.path.join(HERE, 'benchmark_glyph_chart.png'))


def _plot(results, sizes, mode_labels, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not available — skipping chart")
        return

    # Throughput (Mpts/s) per glyph × mode, one chart per N
    glyph_labels = [label for label, *_ in SUITE]
    n_glyphs  = len(glyph_labels)
    n_modes   = len(mode_labels)
    n_sizes   = len(sizes)

    # Build lookup: (label, N, mode) -> Mpts/s
    lut = {}
    for label, n, mode, elapsed, mpts in results:
        lut[(label, n, mode)] = mpts

    COLORS = {'CPU': '#4c9be8', 'GPU': '#f0834a'}
    FALLBACK_COLORS = ['#4c9be8', '#f0834a', '#67c97f', '#e86b6b']

    fig, axes = plt.subplots(1, n_sizes, figsize=(7 * n_sizes, 6),
                             facecolor='#1a1a2e', sharey=False)
    if n_sizes == 1:
        axes = [axes]

    for ax, n in zip(axes, sizes):
        ax.set_facecolor('#12122a')
        x = np.arange(n_glyphs)
        bar_w = 0.8 / n_modes
        for mi, mode in enumerate(mode_labels):
            vals = [lut.get((lbl, n, mode), 0) for lbl in glyph_labels]
            color = COLORS.get(mode, FALLBACK_COLORS[mi % len(FALLBACK_COLORS)])
            offset = (mi - (n_modes - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w,
                          label=mode, color=color, alpha=0.88,
                          edgecolor='#ffffff22', linewidth=0.5)

            # Value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max(vals) * 0.01,
                            f'{v:.1f}', ha='center', va='bottom',
                            fontsize=6.5, color='white')

        ax.set_xticks(x)
        ax.set_xticklabels(glyph_labels, rotation=35, ha='right',
                           fontsize=8, color='#dddddd')
        ax.set_title(f'N = {n:,} points', fontsize=11, color='white', pad=8)
        ax.set_ylabel('Throughput  (Mpts/s)', fontsize=9, color='#cccccc')
        ax.tick_params(colors='#aaaaaa')
        ax.spines['bottom'].set_color('#444')
        ax.spines['left'].set_color('#444')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis='y', color='#333355', linewidth=0.5, zorder=0)

    # Shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS.get(m, FALLBACK_COLORS[i]))
               for i, m in enumerate(mode_labels)]
    fig.legend(handles, mode_labels, loc='upper center', ncol=n_modes,
               fontsize=10, framealpha=0.3, labelcolor='white',
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('Glyph Benchmark: Throughput by Type & Footprint Size',
                 fontsize=13, color='white', fontweight='bold', y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"Chart: {out_path}")


if __name__ == '__main__':
    main()
