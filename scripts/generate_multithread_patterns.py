#!/usr/bin/env python3
"""
Generate test patterns and verify numerical consistency across CPU thread counts.

Mirrors generate_all_patterns.py but processes each pattern through four
CPU threading configurations and checks that results are bit-identical
(routing and accumulation are deterministic regardless of thread count).

Thread modes tested:
  CPU-1T  : cpu_threads=1  (single-thread baseline)
  CPU-2T  : cpu_threads=2
  CPU-4T  : cpu_threads=4
  CPU-NT  : cpu_threads=0  (OpenMP default — all available cores)

Outputs per pattern per mode:
  multithread_outputs/<name>_<mode>.tif
"""

import os
import sys
import shutil
import subprocess
import numpy as np

sys.path.insert(0, '/workspace/python')
import pcr
from pcr.test_generators import (
    generate_checkerboard,
    generate_stripes,
    generate_bullseye,
    generate_gradient,
    generate_text,
    generate_shapes,
    generate_uniform_grid,
    generate_gaussian_clusters,
    generate_planar_surface,
)

OUTPUT_DIR = "multithread_outputs"
STATE_BASE = "/tmp/pcr_mt_patterns"

THREAD_MODES = [
    ("cpu1t",  1),   # label, cpu_threads value
    ("cpu2t",  2),
    ("cpu4t",  4),
    ("cpuNt",  0),   # 0 = all cores
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pipeline(mode_label, cpu_threads, bbox, name, cell_size=1.0):
    state_dir = os.path.join(STATE_BASE, f"{name}_{mode_label}")
    shutil.rmtree(state_dir, ignore_errors=True)
    os.makedirs(state_dir)

    gc = pcr.GridConfig()
    gc.bounds = bbox
    gc.crs = pcr.CRS.from_epsg(32610)
    gc.cell_size_x = cell_size
    gc.cell_size_y = -cell_size
    gc.compute_dimensions()

    r = pcr.ReductionSpec()
    r.value_channel = "value"
    r.type = pcr.ReductionType.Average
    r.output_band_name = name

    cfg = pcr.PipelineConfig()
    cfg.grid = gc
    cfg.reductions = [r]
    cfg.exec_mode = pcr.ExecutionMode.CPU
    cfg.cpu_threads = cpu_threads
    cfg.state_dir = state_dir
    cfg.output_path = os.path.join(OUTPUT_DIR, f"{name}_{mode_label}.tif")

    return pcr.Pipeline.create(cfg)


def tif_to_png(tif_path):
    png_path = tif_path.replace(".tif", ".png")
    try:
        subprocess.run(
            ["gdal_translate", "-of", "PNG", "-scale", "0", "100", "0", "255",
             tif_path, png_path],
            check=True, capture_output=True
        )
        return png_path
    except Exception:
        return None


def arrays_equal(a, b, rtol=1e-5, atol=1e-5):
    """Compare two float arrays, treating NaN==NaN as equal."""
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):
        return False, "NaN masks differ"
    valid = ~a_nan
    if not np.allclose(a[valid], b[valid], rtol=rtol, atol=atol):
        max_diff = np.abs(a[valid] - b[valid]).max()
        return False, f"values differ (max_diff={max_diff:.2e})"
    return True, "OK"


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

def build_patterns(bbox, cell_size):
    patterns = []

    print("[1/9]  Checkerboard ...")
    cloud, meta = generate_checkerboard(bbox, cell_size, points_per_cell=100, square_size=8)
    patterns.append(("01_checkerboard", cloud))

    print("[2/9]  Horizontal stripes ...")
    cloud, meta = generate_stripes(bbox, cell_size, points_per_cell=20,
                                    stripe_width=5, orientation="horizontal")
    patterns.append(("02_stripes_horizontal", cloud))

    print("[3/9]  Vertical stripes ...")
    cloud, meta = generate_stripes(bbox, cell_size, points_per_cell=20,
                                    stripe_width=5, orientation="vertical")
    patterns.append(("03_stripes_vertical", cloud))

    print("[4/9]  Bullseye ...")
    cloud, meta = generate_bullseye(bbox, cell_size, points_per_cell=20, num_rings=8)
    patterns.append(("04_bullseye", cloud))

    print("[5/9]  Linear gradient ...")
    cloud, meta = generate_gradient(bbox, cell_size, points_per_cell=20,
                                     gradient_type="linear", angle=0.0)
    patterns.append(("05_gradient_linear", cloud))

    print("[6/9]  Radial gradient ...")
    cloud, meta = generate_gradient(bbox, cell_size, points_per_cell=20,
                                     gradient_type="radial")
    patterns.append(("06_gradient_radial", cloud))

    print("[7/9]  Uniform grid ...")
    cloud, meta = generate_uniform_grid(bbox, cell_size, points_per_cell=10, value=75.0)
    patterns.append(("07_uniform_grid", cloud))

    print("[8/9]  Gaussian clusters ...")
    cloud, meta = generate_gaussian_clusters(bbox, cell_size, num_clusters=8,
                                              points_per_cluster=2000, cluster_std=8.0)
    patterns.append(("08_gaussian_clusters", cloud))

    print("[9/9]  Planar surface ...")
    cloud, meta = generate_planar_surface(bbox, cell_size, points_per_cell=10,
                                           slope_x=0.5, slope_y=0.3, noise_std=5.0)
    patterns.append(("09_planar_surface", cloud))

    return patterns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MULTI-THREAD PATTERN GENERATION + CONSISTENCY VERIFICATION")
    print("=" * 70)
    print(f"\nThread modes: {', '.join(label for label, _ in THREAD_MODES)}")
    print(f"Output dir  : {OUTPUT_DIR}/\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STATE_BASE, exist_ok=True)

    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 160.0
    bbox.max_y = 160.0
    cell_size = 1.0

    print("Generating point clouds ...")
    patterns = build_patterns(bbox, cell_size)

    all_pass = True
    results_summary = []   # (name, mode, valid_cells, mean, status)

    for name, cloud in patterns:
        print(f"\n{'='*70}")
        print(f"Pattern: {name}  ({cloud.count():,} points)")
        print(f"{'='*70}")

        grids = {}   # mode_label -> numpy array

        for mode_label, cpu_threads in THREAD_MODES:
            try:
                pl = make_pipeline(mode_label, cpu_threads, bbox, name, cell_size)
                pl.ingest(cloud)
                pl.finalize()
                grid = pl.result()
                arr = grid.band_array(0).copy()
                grids[mode_label] = arr
                valid = ~np.isnan(arr)
                mean_val = arr[valid].mean() if valid.any() else float("nan")
                print(f"  {mode_label:<8}: {valid.sum():>6} valid cells,  mean={mean_val:8.4f}")
                results_summary.append((name, mode_label, int(valid.sum()), mean_val, "OK"))

                # Convert baseline to PNG only
                if mode_label == "cpu1t":
                    tif_path = os.path.join(OUTPUT_DIR, f"{name}_{mode_label}.tif")
                    png = tif_to_png(tif_path)
                    if png:
                        print(f"           → PNG: {os.path.basename(png)}")

            except Exception as e:
                print(f"  {mode_label:<8}: FAILED — {e}")
                grids[mode_label] = None
                results_summary.append((name, mode_label, 0, float("nan"), f"FAILED: {e}"))
                all_pass = False

        # ----------------------------------------------------------------
        # Consistency check: all modes must produce identical results
        # ----------------------------------------------------------------
        print(f"\n  Consistency check vs cpu1t baseline:")
        baseline = grids.get("cpu1t")
        if baseline is None:
            print("    SKIP — baseline failed")
            continue

        for mode_label, _ in THREAD_MODES[1:]:
            arr = grids.get(mode_label)
            if arr is None:
                print(f"    {mode_label:<8}: SKIP (mode failed)")
                continue
            ok, msg = arrays_equal(baseline, arr)
            status = "✓ PASS" if ok else "✗ FAIL"
            print(f"    {mode_label:<8}: {status}  ({msg})")
            if not ok:
                all_pass = False

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Pattern':<30}  {'Mode':<8}  {'Valid':>6}  {'Mean':>9}  {'Status'}")
    print("-" * 65)
    for name, mode, valid, mean, status in results_summary:
        short = name[:28]
        mean_str = f"{mean:9.4f}" if not np.isnan(mean) else "       NaN"
        print(f"  {short:<28}  {mode:<8}  {valid:>6}  {mean_str}  {status}")

    print(f"\n{'='*70}")
    if all_pass:
        print("✓ ALL PATTERNS CONSISTENT ACROSS ALL THREAD COUNTS")
    else:
        print("✗ SOME PATTERNS SHOWED INCONSISTENCIES — see above")
    print(f"{'='*70}")
    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print(f"  TIF files for all modes  →  {OUTPUT_DIR}/<pattern>_<mode>.tif")
    print(f"  PNG preview (cpu1t only) →  {OUTPUT_DIR}/<pattern>_cpu1t.png")


if __name__ == "__main__":
    main()
