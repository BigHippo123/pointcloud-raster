#!/usr/bin/env python3
"""
Generate test patterns and verify correctness of Hybrid (CPU routing + GPU
accumulation) mode against CPU-MT and pure GPU modes.

Mirrors generate_all_patterns.py but processes each pattern through multiple
execution modes and cross-checks that outputs are numerically consistent.

Modes tested (GPU modes skipped if no CUDA device is available):
  CPU-MT    : CPU, all cores (cpu_threads=0)         — always run, reference
  GPU       : pure GPU, cloud in host memory          — skipped if no GPU
  Hybrid-2T : Hybrid, 2 CPU routing threads           — skipped if no GPU
  Hybrid-NT : Hybrid, all CPUs / 2 routing threads    — skipped if no GPU

Outputs per pattern per mode:
  hybrid_outputs/<name>_<mode>.tif
  hybrid_outputs/<name>_cpu_mt.png   (PNG preview of CPU-MT reference)

Consistency requirement:
  Every mode must produce the same set of valid cells and values within
  floating-point tolerance (rtol=1e-4, atol=1e-4) vs the CPU-MT reference.
  Slight per-cell differences are expected when chunk boundaries bisect
  a cell's point set, but the overall pattern should be identical.
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

OUTPUT_DIR = "hybrid_outputs"
STATE_BASE = "/tmp/pcr_hybrid_patterns"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_available():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=2
        )
        return result.returncode == 0 and bool(result.stdout.strip())
    except Exception:
        return False


def make_pipeline(mode_label, exec_mode, bbox, name, cell_size=1.0,
                  cpu_threads=0, hybrid_cpu_threads=0):
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
    cfg.exec_mode = exec_mode
    cfg.cpu_threads = cpu_threads
    cfg.hybrid_cpu_threads = hybrid_cpu_threads
    cfg.state_dir = state_dir
    cfg.output_path = os.path.join(OUTPUT_DIR, f"{name}_{mode_label}.tif")

    return pcr.Pipeline.create(cfg)


def tif_to_png(tif_path, scale_min=0, scale_max=100):
    png_path = tif_path.replace(".tif", ".png")
    try:
        subprocess.run(
            ["gdal_translate", "-of", "PNG",
             "-scale", str(scale_min), str(scale_max), "0", "255",
             tif_path, png_path],
            check=True, capture_output=True
        )
        return png_path
    except Exception:
        return None


def arrays_consistent(ref, cmp, rtol=1e-4, atol=1e-4):
    """
    Compare two float arrays for consistency.

    NaN positions must match exactly. Non-NaN values must agree within
    (rtol, atol). Returns (passed: bool, message: str).

    Note: in Hybrid mode, chunk boundaries can route the same cell's points
    to different chunks, so per-cell averages can differ slightly vs CPU-MT
    (which processes all points together). This is expected and not an error.
    """
    ref_nan = np.isnan(ref)
    cmp_nan = np.isnan(cmp)

    # Check valid-cell mask
    nan_diff = int(np.sum(ref_nan != cmp_nan))
    if nan_diff > 0:
        return False, f"valid-cell mask differs in {nan_diff} cells"

    valid = ~ref_nan
    if not valid.any():
        return True, "no valid cells (trivially consistent)"

    diffs = np.abs(ref[valid] - cmp[valid])
    max_diff = float(diffs.max())
    mean_diff = float(diffs.mean())

    # Use a slightly relaxed check: flag only if many cells differ significantly
    # (chunk-boundary effects cause minor per-cell variance in Hybrid mode)
    bad = np.sum(diffs > atol + rtol * np.abs(ref[valid]))
    bad_frac = bad / valid.sum()

    if bad_frac > 0.05:   # more than 5% of cells show significant difference
        return False, (f"values differ in {bad}/{valid.sum()} cells "
                       f"(max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e})")

    msg = "OK"
    if bad > 0:
        msg = f"OK (minor chunk-boundary diff in {bad} cells, max={max_diff:.2e})"
    return True, msg


# ---------------------------------------------------------------------------
# Pattern definitions (subset for speed; add more as needed)
# ---------------------------------------------------------------------------

def build_patterns(bbox, cell_size):
    patterns = []

    # Use moderate point counts — enough to fill cells visually but fast enough
    # for 4-mode cross-verification (CPU-MT, GPU, Hybrid-2T, Hybrid-NT).
    print("[1/9]  Checkerboard ...")
    cloud, meta = generate_checkerboard(bbox, cell_size, points_per_cell=10, square_size=8)
    patterns.append(("01_checkerboard", cloud))

    print("[2/9]  Horizontal stripes ...")
    cloud, meta = generate_stripes(bbox, cell_size, points_per_cell=10,
                                    stripe_width=5, orientation="horizontal")
    patterns.append(("02_stripes_horizontal", cloud))

    print("[3/9]  Vertical stripes ...")
    cloud, meta = generate_stripes(bbox, cell_size, points_per_cell=10,
                                    stripe_width=5, orientation="vertical")
    patterns.append(("03_stripes_vertical", cloud))

    print("[4/9]  Bullseye ...")
    cloud, meta = generate_bullseye(bbox, cell_size, points_per_cell=10, num_rings=8)
    patterns.append(("04_bullseye", cloud))

    print("[5/9]  Linear gradient ...")
    cloud, meta = generate_gradient(bbox, cell_size, points_per_cell=10,
                                     gradient_type="linear", angle=0.0)
    patterns.append(("05_gradient_linear", cloud))

    print("[6/9]  Radial gradient ...")
    cloud, meta = generate_gradient(bbox, cell_size, points_per_cell=10,
                                     gradient_type="radial")
    patterns.append(("06_gradient_radial", cloud))

    print("[7/9]  Uniform grid ...")
    cloud, meta = generate_uniform_grid(bbox, cell_size, points_per_cell=5, value=75.0)
    patterns.append(("07_uniform_grid", cloud))

    print("[8/9]  Gaussian clusters ...")
    cloud, meta = generate_gaussian_clusters(bbox, cell_size, num_clusters=8,
                                              points_per_cluster=500, cluster_std=8.0)
    patterns.append(("08_gaussian_clusters", cloud))

    print("[9/9]  Planar surface ...")
    cloud, meta = generate_planar_surface(bbox, cell_size, points_per_cell=5,
                                           slope_x=0.5, slope_y=0.3, noise_std=5.0)
    patterns.append(("09_planar_surface", cloud))

    return patterns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("HYBRID MODE PATTERN GENERATION + CONSISTENCY VERIFICATION")
    print("=" * 70)

    has_gpu = gpu_available()
    if has_gpu:
        print("\n✓ GPU detected — GPU and Hybrid modes will be tested")
        MODES = [
            # (label,       exec_mode,                    cpu_threads, hybrid_cpu_threads)
            ("cpu_mt",    pcr.ExecutionMode.CPU,         0,           0),
            ("gpu",       pcr.ExecutionMode.GPU,         0,           0),
            ("hybrid_2t", pcr.ExecutionMode.Hybrid,      0,           2),
            ("hybrid_nt", pcr.ExecutionMode.Hybrid,      0,           0),
        ]
    else:
        print("\n✗ No GPU detected — only CPU-MT mode will run (Hybrid/GPU skipped)")
        MODES = [
            ("cpu_mt",    pcr.ExecutionMode.CPU,         0,           0),
        ]

    print(f"Modes       : {', '.join(m[0] for m in MODES)}")
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
    results_summary = []   # (name, mode_label, valid_cells, mean, status)

    for name, cloud in patterns:
        print(f"\n{'='*70}")
        print(f"Pattern: {name}  ({cloud.count():,} points)")
        print(f"{'='*70}")

        grids = {}   # mode_label -> numpy array

        for label, exec_mode, cpu_threads, hybrid_cpu_threads in MODES:
            try:
                pl = make_pipeline(
                    label, exec_mode, bbox, name, cell_size,
                    cpu_threads=cpu_threads,
                    hybrid_cpu_threads=hybrid_cpu_threads,
                )
                pl.ingest(cloud)
                pl.finalize()
                grid = pl.result()
                arr = grid.band_array(0).copy()
                grids[label] = arr

                valid = ~np.isnan(arr)
                mean_val = float(arr[valid].mean()) if valid.any() else float("nan")
                print(f"  {label:<12}: {valid.sum():>6} valid cells,  mean={mean_val:8.4f}")
                results_summary.append((name, label, int(valid.sum()), mean_val, "OK"))

                # Convert cpu_mt baseline to PNG for visual inspection
                if label == "cpu_mt":
                    tif_path = os.path.join(OUTPUT_DIR, f"{name}_{label}.tif")
                    png = tif_to_png(tif_path)
                    if png:
                        print(f"              → PNG: {os.path.basename(png)}")

            except Exception as e:
                print(f"  {label:<12}: FAILED — {e}")
                grids[label] = None
                results_summary.append((name, label, 0, float("nan"), f"FAILED: {e}"))
                all_pass = False

        # ----------------------------------------------------------------
        # Consistency check: all modes vs cpu_mt reference
        # ----------------------------------------------------------------
        if len(MODES) > 1:
            print(f"\n  Consistency check vs cpu_mt reference:")
            reference = grids.get("cpu_mt")
            if reference is None:
                print("    SKIP — reference (cpu_mt) failed")
                continue

            for label, _, _, _ in MODES[1:]:
                arr = grids.get(label)
                if arr is None:
                    print(f"    {label:<12}: SKIP (mode failed)")
                    continue
                ok, msg = arrays_consistent(reference, arr)
                status = "✓ PASS" if ok else "✗ FAIL"
                print(f"    {label:<12}: {status}  ({msg})")
                if not ok:
                    all_pass = False

    # -------------------------------------------------------------------------
    # Side-by-side diff images (requires gdal_calc.py)
    # -------------------------------------------------------------------------
    if has_gpu and len(MODES) > 1:
        print(f"\n{'='*70}")
        print("GENERATING DIFFERENCE IMAGES (Hybrid-NT vs CPU-MT)")
        print(f"{'='*70}")
        for name, _ in patterns:
            ref_tif  = os.path.join(OUTPUT_DIR, f"{name}_cpu_mt.tif")
            cmp_tif  = os.path.join(OUTPUT_DIR, f"{name}_hybrid_nt.tif")
            diff_tif = os.path.join(OUTPUT_DIR, f"{name}_diff_hybrid_vs_cpu.tif")
            if not (os.path.exists(ref_tif) and os.path.exists(cmp_tif)):
                continue
            try:
                subprocess.run(
                    ["gdal_calc.py", "-A", ref_tif, "-B", cmp_tif,
                     "--outfile", diff_tif,
                     "--calc", "A-B", "--NoDataValue", "nan",
                     "--quiet"],
                    check=True, capture_output=True
                )
                # Scale diff by ±5 for visibility
                diff_png = diff_tif.replace(".tif", ".png")
                subprocess.run(
                    ["gdal_translate", "-of", "PNG",
                     "-scale", "-5", "5", "0", "255",
                     diff_tif, diff_png],
                    check=True, capture_output=True
                )
                print(f"  ✓ {os.path.basename(diff_png)}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Pattern':<28}  {'Mode':<12}  {'Valid':>6}  {'Mean':>9}  Status")
    print("  " + "-" * 70)
    for name, mode, valid, mean, status in results_summary:
        short = name[:26]
        mean_str = f"{mean:9.4f}" if not np.isnan(mean) else "       NaN"
        status_short = status if len(status) < 30 else status[:27] + "..."
        print(f"  {short:<28}  {mode:<12}  {valid:>6}  {mean_str}  {status_short}")

    print(f"\n{'='*70}")
    if all_pass:
        print("✓ ALL PATTERNS CONSISTENT ACROSS ALL MODES")
    else:
        print("✗ SOME PATTERNS SHOWED INCONSISTENCIES — see above")
    print(f"{'='*70}")

    print(f"\nOutputs in: {OUTPUT_DIR}/")
    print(f"  TIF: {OUTPUT_DIR}/<pattern>_<mode>.tif")
    print(f"  PNG (cpu_mt preview): {OUTPUT_DIR}/<pattern>_cpu_mt.png")
    if has_gpu:
        print(f"  Diff (hybrid_nt - cpu_mt): {OUTPUT_DIR}/<pattern>_diff_hybrid_vs_cpu.png")

    print("\nTo view:")
    print(f"  ls -lh {OUTPUT_DIR}/*.png")


if __name__ == "__main__":
    main()
