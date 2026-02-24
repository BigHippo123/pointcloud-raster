#!/usr/bin/env python3
"""
DC LiDAR real-data benchmark with comprehensive execution mode and glyph testing.

Reads LAS/LAZ files using the laspy library, streams them one tile at a time
into pipelines, and produces detailed benchmarks showing:
  - Timing breakdown: I/O vs library processing
  - Execution mode comparison: CPU-1T, CPU-MT, GPU, Hybrid
  - Glyph type comparison: Point, Line, Gaussian variants
  - CSV export for systematic analysis

Supports both uncompressed LAS and compressed LAZ files.

Usage:
    python3 scripts/test_dc_lidar.py                    # all files, GPU auto
    python3 scripts/test_dc_lidar.py --subset 3         # quick 3-tile test
    python3 scripts/test_dc_lidar.py --mode all --csv results.csv
    python3 scripts/test_dc_lidar.py --glyph all --mode cpu-mt
    python3 scripts/test_dc_lidar.py --mode gpu --value intensity
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import laspy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import pcr

HERE    = os.path.dirname(os.path.abspath(__file__))
REPO    = os.path.dirname(HERE)
LAS_DIR = os.path.join(REPO, 'glyph_pattern_outputs', 'dc-lidar')
OUT_DIR = os.path.join(REPO, 'glyph_pattern_outputs')

# Glyph configurations for comparison testing
GLYPH_CONFIGS = {
    'point':       {'type': 'point'},
    'line-2':      {'type': 'line', 'half_length': 2.0},
    'line-5':      {'type': 'line', 'half_length': 5.0},
    'gaussian-1':  {'type': 'gaussian', 'sigma': 1.0},
    'gaussian-3':  {'type': 'gaussian', 'sigma': 3.0},
    'gaussian-5':  {'type': 'gaussian', 'sigma': 5.0},
    'gaussian-10': {'type': 'gaussian', 'sigma': 10.0},
}

# ---------------------------------------------------------------------------
# LAS/LAZ reader using laspy library
# ---------------------------------------------------------------------------

def read_las_header(path: str) -> dict:
    """Read LAS/LAZ header using laspy library."""
    with laspy.open(path) as las:
        hdr = las.header
        return {
            'path':    path,
            'scale_x': hdr.scale[0],
            'scale_y': hdr.scale[1],
            'scale_z': hdr.scale[2],
            'npts':    hdr.point_count,
            'min_x':   hdr.mins[0],
            'max_x':   hdr.maxs[0],
            'min_y':   hdr.mins[1],
            'max_y':   hdr.maxs[1],
            'min_z':   hdr.mins[2],
            'max_z':   hdr.maxs[2],
            'size_mb': os.path.getsize(path) / 1e6,
        }


def read_las_points(hdr: dict, ground_only=False) -> dict:
    """Load point data from a LAS/LAZ file using laspy library."""
    las = laspy.read(hdr['path'])

    # Apply ground-only filter if requested
    if ground_only:
        mask = las.classification == 2
        x = np.array(las.x[mask], dtype=np.float64)
        y = np.array(las.y[mask], dtype=np.float64)
        z = np.array(las.z[mask], dtype=np.float64)
        intensity = np.array(las.intensity[mask], dtype=np.float32)
        n_kept = np.sum(mask)
    else:
        x = np.array(las.x, dtype=np.float64)
        y = np.array(las.y, dtype=np.float64)
        z = np.array(las.z, dtype=np.float64)
        intensity = np.array(las.intensity, dtype=np.float32)
        n_kept = len(las.points)

    return {
        'x':         x,
        'y':         y,
        'z':         z,
        'intensity': intensity,
        'n_raw':     hdr['npts'],
        'n_kept':    n_kept,
    }


def scan_directory(las_dir: str, files=None, subset=None) -> list:
    """Read all LAS/LAZ headers; return sorted list of header dicts."""
    if files is None:
        files = sorted(f for f in os.listdir(las_dir)
                      if f.endswith('.las') or f.endswith('.laz'))
    if subset is not None:
        files = files[:subset]
    headers = []
    for fname in files:
        path = os.path.join(las_dir, fname)
        h = read_las_header(path)
        h['fname'] = fname
        headers.append(h)
    return headers


def combined_bbox(headers: list, cell_size=1.0):
    """Compute the bounding box that covers all tiles (padded by 1 cell)."""
    min_x = min(h['min_x'] for h in headers) - cell_size
    max_x = max(h['max_x'] for h in headers) + cell_size
    min_y = min(h['min_y'] for h in headers) - cell_size
    max_y = max(h['max_y'] for h in headers) + cell_size
    return min_x, min_y, max_x, max_y


# ---------------------------------------------------------------------------
# Grid + pipeline helpers
# ---------------------------------------------------------------------------

def make_grid(min_x, min_y, max_x, max_y, cell_size=1.0):
    bbox = pcr.BBox()
    bbox.min_x, bbox.min_y = min_x, min_y
    bbox.max_x, bbox.max_y = max_x, max_y
    gc = pcr.GridConfig()
    gc.bounds      = bbox
    gc.cell_size_x =  cell_size
    gc.cell_size_y = -cell_size   # north-up
    gc.compute_dimensions()
    return gc


def make_cloud_from_data(data: dict, glyph_config: dict, cell_size: float) -> pcr.PointCloud:
    """Create a PointCloud with appropriate channels for the given glyph type."""
    n = data['n_kept']
    cloud = pcr.PointCloud.create(n)
    cloud.set_x_array(data['x'])
    cloud.set_y_array(data['y'])

    # Add standard value channels
    cloud.add_channel('elevation', pcr.DataType.Float32)
    cloud.add_channel('intensity', pcr.DataType.Float32)
    cloud.set_channel_array_f32('elevation', data['z'].astype(np.float32))
    cloud.set_channel_array_f32('intensity', data['intensity'])

    glyph_type = glyph_config['type']

    # Add glyph-specific channels
    if glyph_type == 'line':
        # Add random directions (0 to π radians)
        directions = np.random.uniform(0, np.pi, n).astype(np.float32)
        cloud.add_channel('direction', pcr.DataType.Float32)
        cloud.set_channel_array_f32('direction', directions)

        # Add half_length channel (in world units: cells × cell_size)
        half_length = glyph_config['half_length'] * cell_size
        half_lengths = np.full(n, half_length, dtype=np.float32)
        cloud.add_channel('half_length', pcr.DataType.Float32)
        cloud.set_channel_array_f32('half_length', half_lengths)

    elif glyph_type == 'gaussian':
        # Add sigma channel (in world units: sigma_cells × cell_size)
        sigma = glyph_config['sigma'] * cell_size
        sigmas = np.full(n, sigma, dtype=np.float32)
        cloud.add_channel('sigma', pcr.DataType.Float32)
        cloud.set_channel_array_f32('sigma', sigmas)

    # Point glyph needs no additional channels

    return cloud


def make_reduction_spec(glyph_config: dict, value_channel: str, cell_size: float):
    """Create a ReductionSpec for the given glyph type and value channel."""
    glyph_type = glyph_config['type']

    if glyph_type == 'point':
        spec = pcr.ReductionSpec()
        spec.value_channel = value_channel
        spec.type = pcr.ReductionType.Average
        spec.output_band_name = f'{value_channel}_point'

    elif glyph_type == 'line':
        half_length_cells = glyph_config['half_length']
        spec = pcr.line_splat_spec(
            value_channel=value_channel,
            direction_channel='direction',
            half_length_channel='half_length',
            max_radius_cells=half_length_cells * 2.0,
            output_band_name=f'{value_channel}_line_{half_length_cells:.0f}',
        )

    elif glyph_type == 'gaussian':
        sigma_cells = glyph_config['sigma']
        spec = pcr.gaussian_splat_spec(
            value_channel=value_channel,
            sigma_x_channel='sigma',
            sigma_y_channel='sigma',
            max_radius_cells=sigma_cells * 3.0,
            output_band_name=f'{value_channel}_gaussian_{sigma_cells:.0f}',
        )

    else:
        raise ValueError(f"Unknown glyph type: {glyph_type}")

    return spec


def create_pipeline(spec, gc, mode, state_dir, out_path):
    shutil.rmtree(state_dir, ignore_errors=True)
    cfg = pcr.PipelineConfig()
    cfg.grid               = gc
    cfg.reductions         = [spec]
    cfg.exec_mode          = mode
    cfg.gpu_fallback_to_cpu = True
    cfg.state_dir          = state_dir
    cfg.output_path        = out_path
    pipe = pcr.Pipeline.create(cfg)
    if pipe is None:
        raise RuntimeError("Pipeline.create() returned None")
    return pipe


# ---------------------------------------------------------------------------
# Streaming pipeline runner
# ---------------------------------------------------------------------------

def run_streaming(label, spec, headers, gc, mode, state_dir, out_path,
                  ground_only=False, glyph_config=None, cell_size=1.0, verbose=True):
    """
    Stream LAS files one at a time into a single pipeline.
    Returns a stats dict with detailed timing breakdown and the result band array.
    """
    pipe = create_pipeline(spec, gc, mode, state_dir, out_path)

    if glyph_config is None:
        glyph_config = {'type': 'point'}

    # Timing accumulators
    timing = {
        'io_read':       0.0,
        'cloud_create':  0.0,
        'ingest':        0.0,
        'finalize':      0.0,
    }

    total_pts_raw  = 0
    total_pts_kept = 0

    n_files = len(headers)
    wall_t0 = time.perf_counter()

    if verbose:
        print(f"\n  Streaming {n_files} file(s) into [{label}] pipeline ...")

    for idx, hdr in enumerate(headers, 1):
        fname = hdr['fname']

        # I/O: Read points from disk
        t0 = time.perf_counter()
        data = read_las_points(hdr, ground_only=ground_only)
        timing['io_read'] += time.perf_counter() - t0

        total_pts_raw  += data['n_raw']
        total_pts_kept += data['n_kept']

        if data['n_kept'] == 0:
            if verbose:
                print(f"    [{idx}/{n_files}] {fname}  skipped (0 kept)")
            continue

        # Cloud creation: Python object construction
        t0 = time.perf_counter()
        cloud = make_cloud_from_data(data, glyph_config, cell_size)
        timing['cloud_create'] += time.perf_counter() - t0

        # Library processing: Ingest (routing)
        t0 = time.perf_counter()
        pipe.ingest(cloud)
        timing['ingest'] += time.perf_counter() - t0

        if verbose:
            mpts = data['n_kept'] / timing['ingest'] / 1e6 if timing['ingest'] > 0 else 0
            print(f"    [{idx}/{n_files}] {fname}  "
                  f"io {timing['io_read']:.1f}s  ingest {timing['ingest']:.2f}s  "
                  f"{mpts:.1f} Mpts/s  (kept {data['n_kept']:,}/{data['n_raw']:,})",
                  flush=True)

        del cloud, data

    # Library processing: Finalize (accumulation)
    if verbose:
        print(f"  Finalizing ...", end=' ', flush=True)
    t0 = time.perf_counter()
    pipe.finalize()
    timing['finalize'] = time.perf_counter() - t0
    wall_elapsed = time.perf_counter() - wall_t0
    if verbose:
        print(f"{timing['finalize']:.1f}s")

    # Calculate metrics
    library_time = timing['ingest'] + timing['finalize']
    throughput = total_pts_kept / library_time / 1e6 if library_time > 0 else 0

    if verbose:
        print(f"\n  Timing Breakdown:")
        print(f"    I/O Reading:      {timing['io_read']:.1f}s "
              f"({100*timing['io_read']/wall_elapsed:.0f}%)")
        print(f"    Cloud Creation:   {timing['cloud_create']:.1f}s "
              f"({100*timing['cloud_create']/wall_elapsed:.0f}%)")
        print(f"    Library Ingest:   {timing['ingest']:.1f}s "
              f"({100*timing['ingest']/wall_elapsed:.0f}%)")
        print(f"    Library Finalize: {timing['finalize']:.1f}s "
              f"({100*timing['finalize']/wall_elapsed:.0f}%)")
        print(f"    {'─'*40}")
        print(f"    Library Total:    {library_time:.1f}s "
              f"({100*library_time/wall_elapsed:.0f}%)")
        print(f"    Wall Total:       {wall_elapsed:.1f}s\n")
        print(f"  Throughput (library time): {throughput:.1f} Mpts/s")

    # Grab result band
    arr = None
    try:
        result = pipe.result()
        arr = result.band_array(0).copy()
    except Exception as e:
        if verbose:
            print(f"  Warning: could not retrieve result array: {e}")

    stats = {
        'label':            label,
        'n_files':          n_files,
        'pts_raw':          total_pts_raw,
        'pts_kept':         total_pts_kept,
        'ground_only':      ground_only,
        'io_read_s':        round(timing['io_read'],       3),
        'cloud_create_s':   round(timing['cloud_create'],  3),
        'ingest_s':         round(timing['ingest'],        3),
        'finalize_s':       round(timing['finalize'],      3),
        'library_time_s':   round(library_time,            3),
        'wall_s':           round(wall_elapsed,            3),
        'throughput_mpts':  round(throughput,              3),
    }

    return stats, arr


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def array_stats(arr, label):
    """Compute coverage and value stats from a float32 grid array."""
    if arr is None:
        return {}
    valid = arr[~np.isnan(arr)]
    n_total = arr.size
    n_valid = len(valid)
    return {
        'grid_cells':    n_total,
        'covered_cells': n_valid,
        'coverage_pct':  round(100 * n_valid / n_total, 3),
        'value_min':     round(float(valid.min()),  4) if n_valid else None,
        'value_max':     round(float(valid.max()),  4) if n_valid else None,
        'value_mean':    round(float(valid.mean()), 4) if n_valid else None,
        'value_p5':      round(float(np.percentile(valid,  5)), 4) if n_valid else None,
        'value_p95':     round(float(np.percentile(valid, 95)), 4) if n_valid else None,
    }


# ---------------------------------------------------------------------------
# Comparison PNG (downsampled for large rasters)
# ---------------------------------------------------------------------------

def make_comparison_png(pt_arr, gs_arr, sigma, out_png, value_label,
                        max_px=2000):
    """Side-by-side comparison; downsamples if raster exceeds max_px."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping chart")
        return

    if pt_arr is None or gs_arr is None:
        print("  Result arrays unavailable — skipping chart")
        return

    # Downsample so neither dimension exceeds max_px
    h, w = pt_arr.shape
    factor = max(1, int(np.ceil(max(h, w) / max_px)))
    if factor > 1:
        pt_d = pt_arr[::factor, ::factor]
        gs_d = gs_arr[::factor, ::factor]
        print(f"  Downsampled {w}×{h} → {pt_d.shape[1]}×{pt_d.shape[0]}"
              f" (factor {factor}×) for PNG")
    else:
        pt_d, gs_d = pt_arr, gs_arr

    cmap = 'terrain' if 'elevation' in value_label.lower() else 'gray'

    # Colour limits from valid Point cells (2–98th percentile)
    valid_pt = pt_d[~np.isnan(pt_d)]
    if len(valid_pt) == 0:
        print("  No valid cells — skipping chart")
        return
    vmin = float(np.percentile(valid_pt, 2))
    vmax = float(np.percentile(valid_pt, 98))

    def coverage(arr):
        return 100 * np.sum(~np.isnan(arr)) / arr.size

    # Aspect ratio for subplots
    h_d, w_d = pt_d.shape
    fig_w = 18
    fig_h = max(4, fig_w * h_d / (2 * w_d) + 1.5)

    fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h), facecolor='#1a1a2e')
    panels = [
        (pt_d, f"Point glyph\n{coverage(pt_d):.1f}% covered"),
        (gs_d, f"Gaussian  σ={sigma:.0f} cells\n{coverage(gs_d):.1f}% covered"),
    ]

    for ax, (arr, title) in zip(axes, panels):
        ax.set_facecolor('#111122')
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        ax.set_title(title, color='white', fontsize=12, pad=8)
        ax.axis('off')
        cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.85)
        cb.ax.tick_params(colors='#aaa', labelsize=8)
        cb.set_label(value_label, color='#ccc', fontsize=9)

    fig.suptitle(f'DC LiDAR — Point vs Gaussian Glyph  |  {value_label}',
                 fontsize=13, color='white', fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)
    print(f"  Chart: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DC LiDAR benchmark with execution mode and glyph comparison')
    parser.add_argument('--las-dir',  default=LAS_DIR)
    parser.add_argument('--files',    nargs='+', default=None,
                        help='Specific filenames (default: all *.las/*.laz)')
    parser.add_argument('--subset',   type=int,   default=None,
                        help='Use only first N files')
    parser.add_argument('--cell-size', type=float, default=1.0,
                        help='Raster cell size in metres [default: 1.0]')
    parser.add_argument('--value',    choices=['elevation', 'intensity'],
                        default='elevation')
    parser.add_argument('--ground-only', action='store_true',
                        help='Keep only LAS classification==2 (ground)')
    parser.add_argument('--mode',
                        choices=['cpu-1t', 'cpu-mt', 'gpu', 'hybrid', 'all'],
                        default='gpu',
                        help='Execution mode to test (or "all" for comparison)')
    parser.add_argument('--glyph',
                        choices=['point', 'line-2', 'line-5', 'gaussian-1',
                                'gaussian-3', 'gaussian-5', 'gaussian-10', 'all'],
                        default='point',
                        help='Glyph type to test (or "all" for comparison)')
    parser.add_argument('--csv', default=None,
                        help='Save results to CSV file')
    parser.add_argument('--outdir',   default=OUT_DIR)
    args = parser.parse_args()

    # Determine which modes and glyphs to test
    if args.mode == 'all':
        test_modes = ['cpu-1t', 'cpu-mt', 'gpu', 'hybrid']
    else:
        test_modes = [args.mode]

    if args.glyph == 'all':
        test_glyphs = ['point', 'line-2', 'line-5', 'gaussian-1',
                      'gaussian-3', 'gaussian-5', 'gaussian-10']
    else:
        test_glyphs = [args.glyph]

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # Scan headers
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  DC LiDAR Benchmark")
    print(f"{'='*70}")

    print("\nScanning LAS/LAZ headers ...")
    headers = scan_directory(args.las_dir, files=args.files, subset=args.subset)

    total_pts  = sum(h['npts']    for h in headers)
    total_mb   = sum(h['size_mb'] for h in headers)
    min_x, min_y, max_x, max_y = combined_bbox(headers, args.cell_size)

    print(f"  Files         : {len(headers)}")
    print(f"  Total points  : {total_pts:,}")
    print(f"  Total size    : {total_mb:.0f} MB")
    print(f"  X range       : {min_x:.0f} – {max_x:.0f} m")
    print(f"  Y range       : {min_y:.0f} – {max_y:.0f} m")

    gc = make_grid(min_x, min_y, max_x, max_y, args.cell_size)
    grid_cells = gc.width * gc.height
    print(f"  Grid          : {gc.width} × {gc.height} = {grid_cells:,} cells"
          f"  ({grid_cells / 1e6:.1f}M)")
    print(f"  Cell size     : {args.cell_size}m")
    print(f"  Value channel : {args.value}")
    print(f"  Ground only   : {args.ground_only}")
    print(f"  Test modes    : {', '.join(test_modes)}")
    print(f"  Test glyphs   : {', '.join(test_glyphs)}")

    tmpdir = tempfile.mkdtemp(prefix='pcr_dc_lidar_')

    # ------------------------------------------------------------------
    # Run benchmarks
    # ------------------------------------------------------------------
    all_results = []
    baseline_library_time = None

    try:
        for mode_name in test_modes:
            for glyph_name in test_glyphs:
                # Map mode name to ExecutionMode
                exec_mode_map = {
                    'cpu-1t':  (pcr.ExecutionMode.CPU, 1),
                    'cpu-mt':  (pcr.ExecutionMode.CPU, 0),
                    'gpu':     (pcr.ExecutionMode.GPU, 0),
                    'hybrid':  (pcr.ExecutionMode.Hybrid, 0),
                }
                exec_mode, cpu_threads = exec_mode_map[mode_name]

                glyph_config = GLYPH_CONFIGS[glyph_name]

                # Create label
                glyph_label = glyph_name
                if glyph_config['type'] == 'gaussian':
                    glyph_label = f"Gaussian-σ{glyph_config['sigma']:.0f}"
                elif glyph_config['type'] == 'line':
                    glyph_label = f"Line-{glyph_config['half_length']:.0f}"
                else:
                    glyph_label = "Point"

                label = f"{mode_name.upper()} + {glyph_label}"

                print(f"\n{'='*70}")
                print(f"  {label}")
                print(f"{'='*70}")

                # Create reduction spec
                spec = make_reduction_spec(glyph_config, args.value, args.cell_size)

                # Configure pipeline
                state_dir = os.path.join(tmpdir, f'state_{mode_name}_{glyph_name}')
                out_tif = os.path.join(args.outdir,
                                      f'dc_lidar_{mode_name}_{glyph_name}.tif')

                # Create custom PipelineConfig to set cpu_threads
                shutil.rmtree(state_dir, ignore_errors=True)
                cfg = pcr.PipelineConfig()
                cfg.grid = gc
                cfg.reductions = [spec]
                cfg.exec_mode = exec_mode
                cfg.cpu_threads = cpu_threads
                cfg.gpu_fallback_to_cpu = True
                cfg.state_dir = state_dir
                cfg.output_path = out_tif

                # Create and run pipeline manually (instead of using create_pipeline)
                pipe = pcr.Pipeline.create(cfg)
                if pipe is None:
                    raise RuntimeError("Pipeline.create() returned None")

                # Manual streaming loop (simplified version of run_streaming)
                timing = {
                    'io_read':       0.0,
                    'cloud_create':  0.0,
                    'ingest':        0.0,
                    'finalize':      0.0,
                }
                total_pts_kept = 0
                wall_t0 = time.perf_counter()

                for idx, hdr in enumerate(headers, 1):
                    # I/O
                    t0 = time.perf_counter()
                    data = read_las_points(hdr, ground_only=args.ground_only)
                    timing['io_read'] += time.perf_counter() - t0

                    total_pts_kept += data['n_kept']
                    if data['n_kept'] == 0:
                        continue

                    # Cloud creation
                    t0 = time.perf_counter()
                    cloud = make_cloud_from_data(data, glyph_config, args.cell_size)
                    timing['cloud_create'] += time.perf_counter() - t0

                    # Ingest
                    t0 = time.perf_counter()
                    pipe.ingest(cloud)
                    timing['ingest'] += time.perf_counter() - t0

                    del cloud, data

                # Finalize
                t0 = time.perf_counter()
                pipe.finalize()
                timing['finalize'] = time.perf_counter() - t0
                wall_elapsed = time.perf_counter() - wall_t0

                # Calculate metrics
                library_time = timing['ingest'] + timing['finalize']
                throughput = total_pts_kept / library_time / 1e6 if library_time > 0 else 0

                # Set baseline (CPU-MT + Point)
                if mode_name == 'cpu-mt' and glyph_name == 'point':
                    baseline_library_time = library_time

                # Calculate speedup
                if baseline_library_time and baseline_library_time > 0:
                    speedup = baseline_library_time / library_time
                else:
                    speedup = 1.0

                # Print timing
                print(f"\n  Timing Breakdown:")
                print(f"    I/O Reading:      {timing['io_read']:.1f}s "
                      f"({100*timing['io_read']/wall_elapsed:.0f}%)")
                print(f"    Cloud Creation:   {timing['cloud_create']:.1f}s "
                      f"({100*timing['cloud_create']/wall_elapsed:.0f}%)")
                print(f"    Library Ingest:   {timing['ingest']:.1f}s "
                      f"({100*timing['ingest']/wall_elapsed:.0f}%)")
                print(f"    Library Finalize: {timing['finalize']:.1f}s "
                      f"({100*timing['finalize']/wall_elapsed:.0f}%)")
                print(f"    {'─'*50}")
                print(f"    Library Total:    {library_time:.1f}s "
                      f"({100*library_time/wall_elapsed:.0f}%)")
                print(f"    Wall Total:       {wall_elapsed:.1f}s\n")
                print(f"  Throughput (library time): {throughput:.1f} Mpts/s")
                print(f"  Speedup vs baseline:       {speedup:.2f}x")
                print(f"  → {out_tif}")

                # Store results
                result = {
                    'mode':            mode_name,
                    'glyph':           glyph_name,
                    'glyph_label':     glyph_label,
                    'value_channel':   args.value,
                    'points':          total_pts_kept,
                    'files':           len(headers),
                    'wall_total_s':    round(wall_elapsed,           3),
                    'io_read_s':       round(timing['io_read'],      3),
                    'cloud_create_s':  round(timing['cloud_create'], 3),
                    'ingest_s':        round(timing['ingest'],       3),
                    'finalize_s':      round(timing['finalize'],     3),
                    'library_time_s':  round(library_time,           3),
                    'throughput_mpts': round(throughput,             3),
                    'speedup':         round(speedup,                2),
                    'output_tif':      out_tif,
                }
                all_results.append(result)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"  BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  Files:  {len(headers)}")
    print(f"  Points: {total_pts:,}")
    print(f"  Grid:   {gc.width} × {gc.height} ({grid_cells/1e6:.1f}M cells)\n")

    if len(all_results) > 1:
        # Print comparison table
        print(f"{'Mode':<12} {'Glyph':<15} {'Wall(s)':<8} {'I/O(s)':<8} "
              f"{'Library(s)':<11} {'Mpts/s':<8} {'Speedup':<8}")
        print(f"{'-'*70}")
        for r in all_results:
            print(f"{r['mode']:<12} {r['glyph_label']:<15} "
                  f"{r['wall_total_s']:<8.1f} {r['io_read_s']:<8.1f} "
                  f"{r['library_time_s']:<11.1f} {r['throughput_mpts']:<8.1f} "
                  f"{r['speedup']:<8.2f}x")
        print(f"{'-'*70}")
        print(f"Baseline: CPU-MT + Point (speedup = library_time_baseline / library_time_current)")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------
    if args.csv:
        csv_path = args.csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'mode', 'glyph', 'glyph_label', 'value_channel', 'points', 'files',
                'wall_total_s', 'io_read_s', 'cloud_create_s', 'ingest_s',
                'finalize_s', 'library_time_s', 'throughput_mpts', 'speedup',
                'output_tif'
            ])
            writer.writeheader()
            for r in all_results:
                writer.writerow(r)
        print(f"\n  Results saved to: {csv_path}")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
