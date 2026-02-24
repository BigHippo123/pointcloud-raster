#!/usr/bin/env python3
"""Quick glyph benchmark — completes in ~60 seconds."""

import os, sys, time, shutil, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import pcr

# ---------------------------------------------------------------------------
# Config — edit these to adjust scope
# ---------------------------------------------------------------------------
SIZES   = [100_000, 1_000_000]
GRID    = 500          # 500×500 = 250k cells
REPEATS = 2            # timed runs per cell (best-of)

# (label, reduction_type, glyph_type, footprint_param, max_radius)
# footprint_param = sigma for Gaussian, half_length for Line, None for Point
CASES = [
    ("Point",       pcr.ReductionType.Average,         pcr.GlyphType.Point,    None,  0),
    ("Line hl=2",   pcr.ReductionType.WeightedAverage, pcr.GlyphType.Line,     2,     4),
    ("Line hl=8",   pcr.ReductionType.WeightedAverage, pcr.GlyphType.Line,     8,    10),
    ("Gauss σ=1",   pcr.ReductionType.WeightedAverage, pcr.GlyphType.Gaussian, 1,     4),
    ("Gauss σ=4",   pcr.ReductionType.WeightedAverage, pcr.GlyphType.Gaussian, 4,    16),
    ("Gauss σ=8",   pcr.ReductionType.WeightedAverage, pcr.GlyphType.Gaussian, 8,    32),
]
# ---------------------------------------------------------------------------

def make_spec(reduction_type, glyph_type, param, max_r):
    s = pcr.ReductionSpec()
    s.value_channel = 'value'
    s.type = reduction_type
    s.output_band_name = 'out'
    s.glyph.type = glyph_type
    if glyph_type == pcr.GlyphType.Line:
        s.glyph.default_half_length = float(param)
        s.glyph.default_direction   = 0.5
        s.glyph.max_radius_cells    = float(max_r)
    elif glyph_type == pcr.GlyphType.Gaussian:
        s.glyph.default_sigma_x   = float(param)
        s.glyph.default_sigma_y   = float(param)
        s.glyph.max_radius_cells  = float(max_r)
    return s

def run(n, spec, gc, mode, tmpdir):
    cfg = pcr.PipelineConfig()
    cfg.grid = gc
    cfg.reductions = [spec]
    cfg.exec_mode = mode
    cfg.gpu_fallback_to_cpu = True
    cfg.state_dir = os.path.join(tmpdir, 'state')
    cfg.output_path = os.path.join(tmpdir, 'out.tif')
    pipe = pcr.Pipeline.create(cfg)
    if pipe is None:
        return None
    best = float('inf')
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        pipe.ingest(cloud)
        pipe.finalize()
        best = min(best, time.perf_counter() - t0)
    return best

rng  = np.random.default_rng(42)
bbox = pcr.BBox(); bbox.min_x=0; bbox.min_y=0; bbox.max_x=GRID; bbox.max_y=GRID
gc   = pcr.GridConfig()
gc.bounds = bbox; gc.cell_size_x=1.0; gc.cell_size_y=-1.0; gc.compute_dimensions()
tmpdir = tempfile.mkdtemp()

MODES = [("CPU", pcr.ExecutionMode.CPU), ("GPU", pcr.ExecutionMode.GPU)]

print(f"\nGrid {gc.width}×{gc.height}  |  repeats={REPEATS}  |  best-of reported")
print(f"{'Case':<14} {'N':>9}  {'CPU s':>7}  {'CPU Mpt/s':>10}  {'GPU s':>7}  {'GPU Mpt/s':>10}")
print("─" * 68)

t_overall = time.perf_counter()
results = []

for n in SIZES:
    cloud = pcr.PointCloud.create(n)
    xs = rng.uniform(2, GRID-2, n); ys = rng.uniform(2, GRID-2, n)
    cloud.set_x_array(xs); cloud.set_y_array(ys)
    cloud.add_channel('value', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value', rng.uniform(0,1,n).astype(np.float32))

    for label, rtype, gtype, param, max_r in CASES:
        spec = make_spec(rtype, gtype, param, max_r)
        row = [label, n]
        for mode_name, exec_mode in MODES:
            print(f"  running {label} / {mode_name} / n={n:,} ...", end='\r', flush=True)
            t = run(n, spec, gc, exec_mode, tmpdir)
            if t is None:
                row += [None, None]
            else:
                row += [t, n/t/1e6]
                results.append((label, n, mode_name, t, n/t/1e6))

        cpu_t, cpu_m, gpu_t, gpu_m = row[2], row[3], row[4], row[5]
        ct = f"{cpu_t:7.3f}" if cpu_t else "   N/A "
        cm = f"{cpu_m:10.2f}" if cpu_m else "       N/A"
        gt = f"{gpu_t:7.3f}" if gpu_t else "   N/A "
        gm = f"{gpu_m:10.2f}" if gpu_m else "       N/A"
        print(f"  {' '*50}", end='\r')
        print(f"{label:<14} {n:>9,}  {ct}  {cm}  {gt}  {gm}")

    print("─" * 68)

shutil.rmtree(tmpdir, ignore_errors=True)
elapsed = time.perf_counter() - t_overall
print(f"\nDone in {elapsed:.1f}s")

# --- Chart ---
try:
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    labels    = [c[0] for c in CASES]
    fig, axes = plt.subplots(1, len(SIZES), figsize=(7*len(SIZES), 5),
                             facecolor='#1a1a2e')
    if len(SIZES) == 1: axes = [axes]

    lut = {(lbl,n,m): mpts for lbl,n,m,_,mpts in results}
    colors = {'CPU':'#4c9be8', 'GPU':'#f0834a'}

    for ax, n in zip(axes, SIZES):
        ax.set_facecolor('#12122a')
        x  = np.arange(len(labels))
        bw = 0.35
        for i, (mname, _) in enumerate(MODES):
            vals = [lut.get((lbl, n, mname), 0) for lbl in labels]
            off  = (i - 0.5) * bw
            bars = ax.bar(x+off, vals, bw, label=mname,
                          color=colors[mname], alpha=0.88, edgecolor='#ffffff22')
            for b, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(b.get_x()+b.get_width()/2, b.get_height()*1.02,
                            f'{v:.1f}', ha='center', va='bottom',
                            fontsize=7, color='white')

        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha='right',
                                              fontsize=9, color='#ddd')
        ax.set_title(f'N = {n:,}', fontsize=11, color='white')
        ax.set_ylabel('Mpts/s', color='#ccc'); ax.tick_params(colors='#aaa')
        for sp in ['top','right']: ax.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax.spines[sp].set_color('#444')
        ax.grid(axis='y', color='#2a2a4a', zorder=0)
        ax.legend(fontsize=9, framealpha=0.3, labelcolor='white')

    fig.suptitle('Glyph Throughput: CPU vs GPU', fontsize=13,
                 color='white', fontweight='bold')
    fig.tight_layout()
    chart = os.path.join(os.path.dirname(__file__), 'bench_glyphs_chart.png')
    fig.savefig(chart, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"Chart: {chart}")
except Exception as e:
    print(f"Chart skipped: {e}")
