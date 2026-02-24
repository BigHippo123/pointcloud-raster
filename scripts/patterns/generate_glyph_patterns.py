#!/usr/bin/env python3
"""
Generate visual comparison patterns for Point / Line / Gaussian glyphs.

Produces PNGs in glyph_pattern_outputs/ showing:
  - Gap-filling effect of each glyph type
  - Sigma size progression for Gaussian
  - Direction variations for Line
  - Anisotropic / rotated Gaussian ellipses
  - Adaptive per-point sigma / half_length from channels

Usage:
    python3 scripts/generate_glyph_patterns.py [--mode cpu|gpu]
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import pcr

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glyph_pattern_outputs')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bbox(x0, y0, x1, y1):
    b = pcr.BBox()
    b.min_x, b.min_y, b.max_x, b.max_y = x0, y0, x1, y1
    return b


def make_gc(bbox, cell_size=1.0, tile_size=512):
    gc = pcr.GridConfig()
    gc.bounds = bbox
    gc.cell_size_x = cell_size
    gc.cell_size_y = -cell_size
    gc.tile_width = tile_size
    gc.tile_height = tile_size
    gc.compute_dimensions()
    return gc


def rasterize(cloud, gc, spec, exec_mode, tmpdir):
    """Run a single-reduction pipeline and return the output band as np array."""
    cfg = pcr.PipelineConfig()
    cfg.grid = gc
    cfg.reductions = [spec]
    cfg.exec_mode = exec_mode
    cfg.gpu_fallback_to_cpu = True
    cfg.state_dir = os.path.join(tmpdir, 'state')
    cfg.output_path = os.path.join(tmpdir, 'out.tif')

    # Clean state dir between runs
    if os.path.exists(cfg.state_dir):
        shutil.rmtree(cfg.state_dir)

    pipe = pcr.Pipeline.create(cfg)
    if pipe is None:
        raise RuntimeError("Pipeline.create() returned None — check reduction type is registered")
    pipe.ingest(cloud)
    pipe.finalize()

    return pipe.result().band_array(0).copy()


def save_figure(fig, name):
    path = os.path.join(OUT_DIR, name + '.png')
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


def render_panel(ax, arr, title, cmap='plasma', vmin=None, vmax=None, point_xy=None):
    """Render one raster panel with consistent styling."""
    valid = ~np.isnan(arr)
    if vmin is None:
        vmin = arr[valid].min() if valid.any() else 0
    if vmax is None:
        vmax = arr[valid].max() if valid.any() else 1

    # Prevent collapsed colormap when all values are identical
    if abs(vmax - vmin) < 1e-9:
        vmin, vmax = vmin - 0.5, vmax + 0.5

    arr_display = arr.copy()
    arr_display[~valid] = np.nan

    cmap_obj = plt.colormaps[cmap].copy()
    cmap_obj.set_bad('#0d0d1a')  # dark background for NaN

    im = ax.imshow(arr_display, cmap=cmap_obj, vmin=vmin, vmax=vmax,
                   interpolation='nearest', origin='upper', aspect='equal')

    # Overlay point positions if provided (already in pixel coords)
    if point_xy is not None:
        xs, ys = point_xy
        ax.scatter(xs, ys, s=4, c='white', alpha=0.6, linewidths=0, zorder=3)

    non_nan = valid.sum()
    pct = 100 * non_nan / arr.size
    full_title = f"{title}\n{pct:.0f}% covered"
    ax.set_title(full_title, fontsize=8, color='white', pad=3)
    ax.axis('off')

    return im


def make_sparse_cloud(n, bbox, rng, value_fn=None, add_channels=True):
    """Random sparse point cloud. value_fn(xs, ys) -> float32 array."""
    xs = rng.uniform(bbox.min_x + 1, bbox.max_x - 1, n)
    ys = rng.uniform(bbox.min_y + 1, bbox.max_y - 1, n)
    if value_fn is not None:
        vs = value_fn(xs, ys).astype(np.float32)
    else:
        vs = np.ones(n, dtype=np.float32)

    cloud = pcr.PointCloud.create(n)
    cloud.set_x_array(xs)
    cloud.set_y_array(ys)
    cloud.add_channel('value', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value', vs)
    return cloud, xs, ys


def make_grid_cloud(bbox, step, value_fn=None):
    """Regular grid of points (for showing clear footprints)."""
    xs_1d = np.arange(bbox.min_x + step / 2, bbox.max_x, step)
    ys_1d = np.arange(bbox.min_y + step / 2, bbox.max_y, step)
    xs, ys = np.meshgrid(xs_1d, ys_1d)
    xs, ys = xs.ravel(), ys.ravel()

    if value_fn is not None:
        vs = value_fn(xs, ys).astype(np.float32)
    else:
        vs = np.ones(len(xs), dtype=np.float32)

    cloud = pcr.PointCloud.create(len(xs))
    cloud.set_x_array(xs)
    cloud.set_y_array(ys)
    cloud.add_channel('value', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value', vs)
    return cloud, xs, ys


# ---------------------------------------------------------------------------
# Pattern generators
# ---------------------------------------------------------------------------

def gen_01_gap_fill_comparison(exec_mode, tmpdir, rng):
    """3-column: sparse raw → Gaussian tight → Gaussian wide.  Shows gap fill."""
    print("[01] Gap-fill comparison (Point vs Gaussian sigma=2 vs sigma=6)")
    bbox = make_bbox(0, 0, 120, 120)
    gc = make_gc(bbox, cell_size=1.0)
    n = 1200  # intentionally sparse

    # Gradient surface as values
    def grad(xs, ys):
        cx, cy = 60, 60
        return np.sqrt((xs - cx)**2 + (ys - cy)**2).astype(np.float32)

    cloud, xs, ys = make_sparse_cloud(n, bbox, rng, value_fn=grad)

    # Convert xs,ys to pixel positions for overlay dots
    px = xs - bbox.min_x
    py = (bbox.max_y - ys)

    spec_point = pcr.ReductionSpec()
    spec_point.value_channel = 'value'
    spec_point.type = pcr.ReductionType.Average
    spec_point.output_band_name = 'point'

    spec_g2 = pcr.gaussian_splat_spec('value', default_sigma=2.0, max_radius_cells=8.0)
    spec_g6 = pcr.gaussian_splat_spec('value', default_sigma=6.0, max_radius_cells=20.0)

    arr_pt = rasterize(cloud, gc, spec_point, exec_mode, tmpdir)
    arr_g2 = rasterize(cloud, gc, spec_g2, exec_mode, tmpdir)
    arr_g6 = rasterize(cloud, gc, spec_g6, exec_mode, tmpdir)

    fig = plt.figure(figsize=(15, 5.5), facecolor='#1a1a2e')
    fig.suptitle('Gap-Fill Comparison: Point vs Gaussian Splat', fontsize=13,
                 color='white', fontweight='bold', y=1.01)

    vmin, vmax = 0, 85
    for i, (arr, title) in enumerate([
        (arr_pt, f'Point (Average)\n{n:,} sparse points'),
        (arr_g2, 'Gaussian  σ = 2 cells\n(WeightedAverage)'),
        (arr_g6, 'Gaussian  σ = 6 cells\n(WeightedAverage)'),
    ]):
        ax = fig.add_subplot(1, 3, i + 1)
        # Only overlay points on the first panel, at reduced size so they're visible
        overlay = (px, py) if i == 0 else None
        render_panel(ax, arr, title, cmap='magma', vmin=vmin, vmax=vmax,
                     point_xy=overlay)

    fig.tight_layout(pad=1.2)
    return save_figure(fig, '01_gap_fill_comparison')


def gen_02_sigma_progression(exec_mode, tmpdir, rng):
    """6 panels: σ = 0.5, 1, 2, 4, 8, 16 on a fixed sparse cloud (sinusoidal values)."""
    print("[02] Gaussian sigma progression (σ = 0.5 → 16)")
    bbox = make_bbox(0, 0, 100, 100)
    gc = make_gc(bbox, cell_size=1.0)
    n = 800

    # Sinusoidal surface so smoothing is visible at every sigma
    def checker_wave(xs, ys):
        return (np.sin(xs * np.pi / 12) * np.cos(ys * np.pi / 12) * 0.5 + 0.5).astype(np.float32)

    cloud, xs, ys = make_sparse_cloud(n, bbox, rng, value_fn=checker_wave)
    px = xs - bbox.min_x
    py = bbox.max_y - ys

    sigmas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    arrs = []
    for s in sigmas:
        spec = pcr.gaussian_splat_spec('value', default_sigma=s,
                                       max_radius_cells=max(4 * s, 4))
        arrs.append(rasterize(cloud, gc, spec, exec_mode, tmpdir))

    fig = plt.figure(figsize=(18, 7), facecolor='#1a1a2e')
    fig.suptitle('Gaussian Splat — Sigma Progression  (sinusoidal surface, 800 sparse pts)',
                 fontsize=13, color='white', fontweight='bold', y=1.01)

    for i, (arr, s) in enumerate(zip(arrs, sigmas)):
        ax = fig.add_subplot(2, 3, i + 1)
        # Use per-panel auto scale so internal contrast is always visible,
        # but title shows sigma so you can compare sharpness across panels
        render_panel(ax, arr, f'σ = {s} cells', cmap='inferno',
                     point_xy=(px, py) if i == 0 else None)

    fig.tight_layout(pad=1.2)
    return save_figure(fig, '02_sigma_progression')


def gen_03_anisotropic_gaussian(exec_mode, tmpdir, rng):
    """Show elliptical Gaussian: vary sigma_x vs sigma_y and rotation."""
    print("[03] Anisotropic / rotated Gaussian ellipses")
    bbox = make_bbox(0, 0, 80, 80)
    gc = make_gc(bbox, cell_size=1.0)

    # Single point at center for clean footprint view
    cloud_center = pcr.PointCloud.create(1)
    cloud_center.set_x_array(np.array([40.0]))
    cloud_center.set_y_array(np.array([40.0]))
    cloud_center.add_channel('value', pcr.DataType.Float32)
    cloud_center.set_channel_array_f32('value', np.ones(1, np.float32))

    configs = [
        # (label, sigma_x, sigma_y, rotation_deg)
        ('Isotropic\nσ=4', 4, 4, 0),
        ('Elongated X\nσx=8, σy=2', 8, 2, 0),
        ('Elongated Y\nσx=2, σy=8', 2, 8, 0),
        ('Rotated 45°\nσx=8, σy=2', 8, 2, 45),
        ('Rotated 30°\nσx=10, σy=1.5', 10, 1.5, 30),
        ('Rotated 75°\nσx=10, σy=1.5', 10, 1.5, 75),
    ]

    arrs = []
    for _, sx, sy, rot_deg in configs:
        rot_rad = np.radians(rot_deg)
        cloud_r = pcr.PointCloud.create(1)
        cloud_r.set_x_array(np.array([40.0]))
        cloud_r.set_y_array(np.array([40.0]))
        cloud_r.add_channel('value', pcr.DataType.Float32)
        cloud_r.set_channel_array_f32('value', np.ones(1, np.float32))
        cloud_r.add_channel('rot', pcr.DataType.Float32)
        cloud_r.set_channel_array_f32('rot', np.array([rot_rad], np.float32))

        # Use Sum so the output = value * Gaussian_weight = the actual kernel shape
        spec = pcr.ReductionSpec()
        spec.value_channel = 'value'
        spec.type = pcr.ReductionType.Sum
        spec.output_band_name = 'g'
        spec.glyph.type = pcr.GlyphType.Gaussian
        spec.glyph.rotation_channel = 'rot'
        spec.glyph.default_sigma_x = float(sx)
        spec.glyph.default_sigma_y = float(sy)
        spec.glyph.max_radius_cells = max(4 * max(sx, sy), 8)
        arrs.append(rasterize(cloud_r, gc, spec, exec_mode, tmpdir))

    fig = plt.figure(figsize=(18, 7), facecolor='#1a1a2e')
    fig.suptitle('Anisotropic Gaussian — sigma_x, sigma_y, rotation', fontsize=13,
                 color='white', fontweight='bold', y=1.01)

    for i, (arr, (label, *_)) in enumerate(zip(arrs, configs)):
        ax = fig.add_subplot(2, 3, i + 1)
        # vmin=0 so NaN (dark) and zero-weight cells are distinguished;
        # vmax=None → auto from per-panel peak (each ellipse has peak ~1)
        render_panel(ax, arr, label, cmap='viridis', vmin=0)

    fig.tight_layout(pad=1.2)
    return save_figure(fig, '03_anisotropic_gaussian')


def gen_04_line_directions(exec_mode, tmpdir, rng):
    """Grid of isolated points each painted as a line glyph at various angles."""
    print("[04] Line glyph — direction sweep")

    # ---- Panel A: 3×3 direction sweep on a sparse background ----
    # Each of 9 lone points at a different angle, all value=1
    bbox9 = make_bbox(0, 0, 100, 100)
    gc9 = make_gc(bbox9, cell_size=1.0)

    directions_deg = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 0]
    centers = [(18 + 32 * (i % 3), 18 + 32 * (i // 3)) for i in range(9)]
    xs9  = np.array([c[0] for c in centers], dtype=np.float64)
    ys9  = np.array([c[1] for c in centers], dtype=np.float64)
    dirs9 = np.array([np.radians(d) for d in directions_deg], dtype=np.float32)

    cloud9 = pcr.PointCloud.create(9)
    cloud9.set_x_array(xs9); cloud9.set_y_array(ys9)
    cloud9.add_channel('value', pcr.DataType.Float32)
    cloud9.set_channel_array_f32('value', np.ones(9, np.float32))
    cloud9.add_channel('dir', pcr.DataType.Float32)
    cloud9.set_channel_array_f32('dir', dirs9)

    spec9 = pcr.line_splat_spec('value', direction_channel='dir',
                                default_half_length=12.0, max_radius_cells=14.0)
    arr9 = rasterize(cloud9, gc9, spec9, exec_mode, tmpdir)

    # ---- Panel B: 6 half-length panels on a zoomed grid ----
    # Small grid so each line segment fills the panel nicely
    bbox1 = make_bbox(0, 0, 70, 70)
    gc1 = make_gc(bbox1, cell_size=1.0)

    cloud1 = pcr.PointCloud.create(1)
    cloud1.set_x_array(np.array([35.0]))
    cloud1.set_y_array(np.array([35.0]))
    cloud1.add_channel('value', pcr.DataType.Float32)
    cloud1.set_channel_array_f32('value', np.ones(1, np.float32))

    half_lengths = [2, 5, 10, 18, 26, 32]
    hl_arrs = []
    for hl in half_lengths:
        spec_hl = pcr.line_splat_spec('value', default_direction=np.radians(30),
                                      default_half_length=float(hl),
                                      max_radius_cells=float(hl) + 2)
        hl_arrs.append(rasterize(cloud1, gc1, spec_hl, exec_mode, tmpdir))

    fig = plt.figure(figsize=(18, 10), facecolor='#1a1a2e')
    fig.suptitle('Line Glyph — Direction Sweep & Half-Length Progression', fontsize=13,
                 color='white', fontweight='bold')

    # Top row: direction sweep
    ax_top = fig.add_subplot(2, 1, 1)
    render_panel(ax_top, arr9, 'Direction sweep: 0° 22.5° 45° … 157.5°  (half_length=12 cells)',
                 cmap='hot')

    # Bottom row: half-length sweep (use Sum so count of cells hit is visible)
    for i, (arr_hl, hl) in enumerate(zip(hl_arrs, half_lengths)):
        ax = fig.add_subplot(2, 6, 7 + i)
        n_cells = int((~np.isnan(arr_hl)).sum())
        render_panel(ax, arr_hl,
                     f'half_length={hl}\n({n_cells} cells)',
                     cmap='hot')

    fig.tight_layout(pad=1.2)
    return save_figure(fig, '04_line_directions')


def gen_05_flow_field(exec_mode, tmpdir, rng):
    """Line glyphs oriented along a simulated 2D vector field (wind / flow)."""
    print("[05] Line glyph — vector flow field")
    bbox = make_bbox(0, 0, 120, 120)
    gc = make_gc(bbox, cell_size=1.0)

    # Regular grid of points for clean visualization
    cloud, xs, ys = make_grid_cloud(bbox, step=8.0)
    n = cloud.count()

    # Simulated flow: counter-clockwise vortex
    cx, cy = 60.0, 60.0
    dx = ys - cy  # perpendicular to radius = tangent direction
    dy = -(xs - cx)
    norm = np.sqrt(dx**2 + dy**2) + 1e-9
    dx_n, dy_n = dx / norm, dy / norm

    directions = np.arctan2(dy_n, dx_n).astype(np.float32)
    # Clamp near-center points to min half_length so they don't vanish
    half_lengths = np.clip(norm / 10 + 2, 2, 10).astype(np.float32)
    # Value = normalised radius (0 at center, 1 at edge) — avoids 0/0 at center
    values = np.clip(norm / norm.max(), 0.01, 1.0).astype(np.float32)

    cloud.add_channel('dir', pcr.DataType.Float32)
    cloud.set_channel_array_f32('dir', directions)
    cloud.add_channel('hl', pcr.DataType.Float32)
    cloud.set_channel_array_f32('hl', half_lengths)
    cloud.set_channel_array_f32('value', values)

    spec = pcr.line_splat_spec('value', direction_channel='dir',
                               half_length_channel='hl',
                               max_radius_cells=12.0)
    arr = rasterize(cloud, gc, spec, exec_mode, tmpdir)

    # Adaptive Gaussian: sigma grows with distance from center
    # Use Sum reduction so kernel weight IS the output value — avoids the
    # WeightedAverage flattening that makes all cells look the same
    sigmas = np.clip(norm / 10 + 0.5, 0.5, 5.0).astype(np.float32)
    cloud.add_channel('sigma', pcr.DataType.Float32)
    cloud.set_channel_array_f32('sigma', sigmas)

    spec_g = pcr.ReductionSpec()
    spec_g.value_channel = 'value'
    spec_g.type = pcr.ReductionType.Sum
    spec_g.output_band_name = 'g'
    spec_g.glyph.type = pcr.GlyphType.Gaussian
    spec_g.glyph.sigma_x_channel = 'sigma'
    spec_g.glyph.sigma_y_channel = 'sigma'
    spec_g.glyph.max_radius_cells = 18.0
    arr_g = rasterize(cloud, gc, spec_g, exec_mode, tmpdir)

    fig = plt.figure(figsize=(14, 6.5), facecolor='#1a1a2e')
    fig.suptitle('Flow Field — Line Glyph vs Adaptive Gaussian', fontsize=13,
                 color='white', fontweight='bold', y=1.01)

    ax1 = fig.add_subplot(1, 2, 1)
    render_panel(ax1, arr, 'Line glyph (vortex field)\nper-point direction + length',
                 cmap='cool')

    ax2 = fig.add_subplot(1, 2, 2)
    render_panel(ax2, arr_g, 'Adaptive Gaussian\nσ ∝ distance from center',
                 cmap='cool')

    fig.tight_layout(pad=1.2)
    return save_figure(fig, '05_flow_field')


def gen_06_sparse_vs_dense(exec_mode, tmpdir, rng):
    """Point vs Gaussian on very sparse vs dense point clouds."""
    print("[06] Density comparison: sparse (50pts) vs medium (500pts) vs dense (5000pts)")
    bbox = make_bbox(0, 0, 100, 100)
    gc = make_gc(bbox, cell_size=1.0)

    def radial(xs, ys):
        r = np.sqrt((xs - 50)**2 + (ys - 50)**2)
        return np.sin(r / 8).astype(np.float32)

    n_list = [50, 500, 5000]
    # Layout: 2 cols (Point | Gaussian), 3 rows (n=50 / 500 / 5000).
    # Make figure tall so each 100×100 cell is rendered square.
    fig = plt.figure(figsize=(10, 15), facecolor='#1a1a2e')
    fig.suptitle('Point vs Gaussian: Sparse → Dense Point Cloud', fontsize=13,
                 color='white', fontweight='bold', y=1.005)

    vmin, vmax = -1.0, 1.0
    for row, n in enumerate(n_list):
        cloud, xs, ys = make_sparse_cloud(n, bbox, rng, value_fn=radial)
        px = xs - bbox.min_x
        py = bbox.max_y - ys

        spec_pt = pcr.ReductionSpec()
        spec_pt.value_channel = 'value'
        spec_pt.type = pcr.ReductionType.Average
        spec_pt.output_band_name = 'pt'

        spec_g = pcr.gaussian_splat_spec('value', default_sigma=2.5, max_radius_cells=10.0)

        arr_pt = rasterize(cloud, gc, spec_pt, exec_mode, tmpdir)
        arr_g  = rasterize(cloud, gc, spec_g,  exec_mode, tmpdir)

        for col, (arr, title) in enumerate([
            (arr_pt, f'Point / n={n:,}'),
            (arr_g,  f'Gaussian σ=2.5 / n={n:,}'),
        ]):
            ax = fig.add_subplot(3, 2, row * 2 + col + 1)
            render_panel(ax, arr, title, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                         point_xy=(px, py) if col == 0 else None)

    fig.tight_layout(pad=0.8)
    return save_figure(fig, '06_sparse_vs_dense')


def gen_07_per_point_sigma(exec_mode, tmpdir, rng):
    """Gaussian with per-point sigma from a channel: large sigma at edges."""
    print("[07] Adaptive per-point sigma (grows from center)")
    bbox = make_bbox(0, 0, 100, 100)
    gc = make_gc(bbox, cell_size=1.0)
    n = 300

    xs = rng.uniform(5, 95, n)
    ys = rng.uniform(5, 95, n)

    cx, cy = 50.0, 50.0
    dist = np.sqrt((xs - cx)**2 + (ys - cy)**2)
    sigma_vals = np.clip(dist / 10.0 + 0.5, 0.5, 6.0).astype(np.float32)

    # Use value = 1.0 and Sum reduction so output = sum of Gaussian weights at each cell.
    # Cells covered by many/large footprints will be brighter — makes the sigma
    # effect visible rather than being cancelled by WeightedAverage.
    vs = np.ones(n, dtype=np.float32)

    cloud = pcr.PointCloud.create(n)
    cloud.set_x_array(xs)
    cloud.set_y_array(ys)
    cloud.add_channel('value', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value', vs)
    cloud.add_channel('sigma', pcr.DataType.Float32)
    cloud.set_channel_array_f32('sigma', sigma_vals)

    def make_gauss_sum_spec(sigma_ch='', default_sigma=2.0, max_r=8.0):
        spec = pcr.ReductionSpec()
        spec.value_channel = 'value'
        spec.type = pcr.ReductionType.Sum
        spec.output_band_name = 'g'
        spec.glyph.type = pcr.GlyphType.Gaussian
        spec.glyph.sigma_x_channel = sigma_ch
        spec.glyph.sigma_y_channel = sigma_ch
        spec.glyph.default_sigma_x = default_sigma
        spec.glyph.default_sigma_y = default_sigma
        spec.glyph.max_radius_cells = max_r
        return spec

    arr_u = rasterize(cloud, gc, make_gauss_sum_spec(default_sigma=2.0, max_r=8.0),
                      exec_mode, tmpdir)
    arr_a = rasterize(cloud, gc, make_gauss_sum_spec(sigma_ch='sigma', max_r=28.0),
                      exec_mode, tmpdir)

    fig = plt.figure(figsize=(16, 5.5), facecolor='#1a1a2e')
    fig.suptitle('Adaptive Per-Point Sigma — σ grows with distance from center\n'
                 '(Sum reduction: brightness = accumulated Gaussian weight)',
                 fontsize=12, color='white', fontweight='bold', y=1.03)

    # Panel 1: sigma scatter
    ax0 = fig.add_subplot(1, 3, 1)
    sc = ax0.scatter(xs - bbox.min_x, bbox.max_y - ys, c=sigma_vals,
                     cmap='plasma', s=10, vmin=0.5, vmax=6.0)
    ax0.set_facecolor('#0d0d1a')
    ax0.set_title('Point cloud\n(color = σ per point)', fontsize=9, color='white')
    ax0.set_xlim(0, gc.width); ax0.set_ylim(gc.height, 0)
    ax0.axis('off')
    cb = plt.colorbar(sc, ax=ax0, fraction=0.046, label='σ (cells)')
    cb.ax.yaxis.label.set_color('white')
    cb.ax.tick_params(colors='white')

    ax1 = fig.add_subplot(1, 3, 2)
    render_panel(ax1, arr_u, 'Uniform σ = 2.0 cells\n(Sum: uniform blob density)',
                 cmap='plasma', vmin=0)

    ax2 = fig.add_subplot(1, 3, 3)
    render_panel(ax2, arr_a, 'Adaptive σ ∝ dist from center\n(edges brighter: larger footprint sums)',
                 cmap='plasma', vmin=0)

    fig.tight_layout(pad=1.2)
    return save_figure(fig, '07_per_point_sigma')


def gen_08_glyph_showcase(exec_mode, tmpdir, rng):
    """Single-page showcase: one structured pattern through all glyph modes."""
    print("[08] Showcase: bullseye pattern through Point / Line / Gaussian")
    bbox = make_bbox(0, 0, 128, 128)
    gc = make_gc(bbox, cell_size=1.0)

    # Concentric ring sparse cloud — like LiDAR scan rings
    n_rings = 8
    n_per_ring = 120
    cx, cy = 64.0, 64.0
    radii = np.linspace(8, 56, n_rings)
    ring_vals = np.linspace(0.1, 1.0, n_rings)

    xs_list, ys_list, vs_list, dirs_list = [], [], [], []
    for r, v in zip(radii, ring_vals):
        angles = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)
        # add small jitter
        angles += rng.uniform(-0.05, 0.05, n_per_ring)
        xs_list.append(cx + r * np.cos(angles))
        ys_list.append(cy + r * np.sin(angles))
        vs_list.append(np.full(n_per_ring, v))
        # tangent direction (perpendicular to radius = scan direction)
        dirs_list.append(angles + np.pi / 2)

    xs = np.concatenate(xs_list)
    ys = np.concatenate(ys_list)
    vs = np.concatenate(vs_list).astype(np.float32)
    dirs = np.concatenate(dirs_list).astype(np.float32)

    cloud = pcr.PointCloud.create(len(xs))
    cloud.set_x_array(xs)
    cloud.set_y_array(ys)
    cloud.add_channel('value', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value', vs)
    cloud.add_channel('dir', pcr.DataType.Float32)
    cloud.set_channel_array_f32('dir', dirs)

    spec_pt = pcr.ReductionSpec()
    spec_pt.value_channel = 'value'
    spec_pt.type = pcr.ReductionType.Average
    spec_pt.output_band_name = 'pt'

    spec_ln = pcr.line_splat_spec('value', direction_channel='dir',
                                   default_half_length=4.0, max_radius_cells=6.0)
    spec_g2 = pcr.gaussian_splat_spec('value', default_sigma=2.0, max_radius_cells=8.0)
    spec_g5 = pcr.gaussian_splat_spec('value', default_sigma=5.0, max_radius_cells=18.0)

    arr_pt = rasterize(cloud, gc, spec_pt, exec_mode, tmpdir)
    arr_ln = rasterize(cloud, gc, spec_ln, exec_mode, tmpdir)
    arr_g2 = rasterize(cloud, gc, spec_g2, exec_mode, tmpdir)
    arr_g5 = rasterize(cloud, gc, spec_g5, exec_mode, tmpdir)

    fig = plt.figure(figsize=(18, 5.5), facecolor='#1a1a2e')
    fig.suptitle('Bullseye Pattern — Point Cloud Rendered with Different Glyph Types',
                 fontsize=13, color='white', fontweight='bold', y=1.01)

    px = xs - bbox.min_x
    py = bbox.max_y - ys

    for i, (arr, title) in enumerate([
        (arr_pt, f'Point (Average)\n{len(xs):,} pts on 8 rings'),
        (arr_ln, 'Line glyph\ntangent direction, half_length=4'),
        (arr_g2, 'Gaussian  σ = 2 cells'),
        (arr_g5, 'Gaussian  σ = 5 cells'),
    ]):
        ax = fig.add_subplot(1, 4, i + 1)
        render_panel(ax, arr, title, cmap='turbo', vmin=0, vmax=1.0,
                     point_xy=(px, py) if i == 0 else None)

    fig.tight_layout(pad=1.2)
    return save_figure(fig, '08_glyph_showcase')


def gen_09_combined_index(paths):
    """Generate an HTML index page."""
    print("[idx] Generating HTML index")
    html_path = os.path.join(OUT_DIR, 'index.html')

    thumbs = '\n'.join(
        f'''    <div style="display:inline-block;margin:8px;text-align:center">
      <a href="{os.path.basename(p)}">
        <img src="{os.path.basename(p)}" style="height:200px;border:1px solid #444"/><br/>
        <span style="color:#ccc;font-size:12px">{os.path.basename(p)}</span>
      </a>
    </div>'''
        for p in paths
    )

    html = f"""<!DOCTYPE html>
<html><head><title>PCR Glyph Patterns</title>
<style>body{{background:#1a1a2e;font-family:sans-serif;color:white;padding:20px}}</style>
</head><body>
<h1>PCR Glyph Pattern Outputs</h1>
<p>Generated by <code>scripts/generate_glyph_patterns.py</code></p>
{thumbs}
</body></html>
"""
    with open(html_path, 'w') as f:
        f.write(html)
    print(f"  Saved: {html_path}")
    return html_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate glyph pattern visuals')
    parser.add_argument('--mode', choices=['cpu', 'gpu'], default='gpu',
                        help='Execution mode (gpu falls back to cpu if unavailable)')
    args = parser.parse_args()

    mode_map = {'cpu': pcr.ExecutionMode.CPU, 'gpu': pcr.ExecutionMode.GPU}
    exec_mode = mode_map[args.mode]

    os.makedirs(OUT_DIR, exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix='pcr_glyph_')
    rng = np.random.default_rng(2025)

    print(f"\nOutput directory: {OUT_DIR}")
    print(f"Execution mode:   {args.mode.upper()} (gpu_fallback_to_cpu=True)")
    print()

    paths = []
    generators = [
        gen_01_gap_fill_comparison,
        gen_02_sigma_progression,
        gen_03_anisotropic_gaussian,
        gen_04_line_directions,
        gen_05_flow_field,
        gen_06_sparse_vs_dense,
        gen_07_per_point_sigma,
        gen_08_glyph_showcase,
    ]

    try:
        for gen_fn in generators:
            p = gen_fn(exec_mode, tmpdir, rng)
            paths.append(p)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    gen_09_combined_index(paths)

    print(f"\n{'='*60}")
    print(f"Done! Generated {len(paths)} pattern images.")
    print(f"Open: {os.path.join(OUT_DIR, 'index.html')}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
