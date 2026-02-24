#!/usr/bin/env python3
"""
Benchmark CPU multi-threading performance across thread counts.

Measures ingest+finalize throughput only (Pipeline.create is excluded from
timing so one-time init cost does not distort per-run results).

Thread configurations tested for each scale:
  CPU-1T  : cpu_threads=1  (single-threaded baseline)
  CPU-2T  : cpu_threads=2
  CPU-4T  : cpu_threads=4
  CPU-NT  : cpu_threads=0  (all available cores, OpenMP default)
  GPU     : GPU mode for reference (if available)
"""

import os
import sys
import time
import shutil
import subprocess
import numpy as np

sys.path.insert(0, '/workspace/python')
import pcr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_available():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=2
        )
        return result.returncode == 0 and result.stdout.strip()
    except Exception:
        return False


def make_random_cloud(n, bbox):
    cloud = pcr.PointCloud.create(n)
    rng = np.random.default_rng(42)
    xs = rng.uniform(bbox.min_x + 0.5, bbox.max_x - 0.5, n).astype(np.float64)
    ys = rng.uniform(bbox.min_y + 0.5, bbox.max_y - 0.5, n).astype(np.float64)
    vs = rng.uniform(0, 100, n).astype(np.float32)
    cloud.set_x_array(xs)
    cloud.set_y_array(ys)
    cloud.add_channel('value', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value', vs)
    return cloud


def build_pipeline(mode, bbox, state_dir, output_path, cpu_threads=0):
    gc = pcr.GridConfig()
    gc.bounds = bbox
    gc.crs = pcr.CRS.from_epsg(32610)
    gc.cell_size_x = 1.0
    gc.cell_size_y = -1.0
    gc.compute_dimensions()

    r = pcr.ReductionSpec()
    r.value_channel = 'value'
    r.type = pcr.ReductionType.Average
    r.output_band_name = 'out'

    cfg = pcr.PipelineConfig()
    cfg.grid = gc
    cfg.reductions = [r]
    cfg.exec_mode = mode
    cfg.cpu_threads = cpu_threads
    cfg.output_path = output_path
    cfg.state_dir = state_dir

    shutil.rmtree(state_dir, ignore_errors=True)
    os.makedirs(state_dir)

    return pcr.Pipeline.create(cfg)


def time_ingest_finalize(pipeline, cloud):
    t0 = time.perf_counter()
    pipeline.ingest(cloud)
    pipeline.finalize()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("CPU Multi-Thread Scaling Benchmark")
    print("=" * 80)

    has_gpu = gpu_available()
    if has_gpu:
        print("\n✓ GPU detected (will include GPU reference row)\n")
    else:
        print("\n✗ GPU not detected (GPU row will be skipped)\n")

    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 160.0
    bbox.max_y = 160.0

    scales = [1_000_000, 5_000_000, 10_000_000, 25_000_000]

    # (n_points, label, thread_count_or_None_for_gpu, elapsed, throughput)
    rows = []

    for n in scales:
        label_n = f"{n:,}"
        print(f"{'='*80}")
        print(f"Scale: {label_n} points")
        print(f"{'='*80}")

        cloud = make_random_cloud(n, bbox)

        configs = [
            ('CPU-1T',  pcr.ExecutionMode.CPU, 1),
            ('CPU-2T',  pcr.ExecutionMode.CPU, 2),
            ('CPU-4T',  pcr.ExecutionMode.CPU, 4),
            ('CPU-NT',  pcr.ExecutionMode.CPU, 0),
        ]

        for label, mode, threads in configs:
            try:
                pl = build_pipeline(
                    mode, bbox,
                    f'/tmp/pcr_mt_{label.lower()}',
                    f'/tmp/bench_mt_{label.lower()}.tif',
                    cpu_threads=threads
                )
                elapsed = time_ingest_finalize(pl, cloud)
                tput = n / elapsed / 1e6
                print(f"  {label:<10}: {elapsed:.3f}s  ({tput:.2f} Mpt/s)")
                rows.append((n, label, elapsed, tput))
            except Exception as e:
                print(f"  {label:<10}: FAILED — {e}")
                rows.append((n, label, None, None))

        # GPU reference row
        if has_gpu:
            try:
                pl = build_pipeline(
                    pcr.ExecutionMode.GPU, bbox,
                    '/tmp/pcr_mt_gpu',
                    '/tmp/bench_mt_gpu.tif',
                    cpu_threads=0
                )
                elapsed = time_ingest_finalize(pl, cloud)
                tput = n / elapsed / 1e6
                print(f"  {'GPU':<10}: {elapsed:.3f}s  ({tput:.2f} Mpt/s)")
                rows.append((n, 'GPU', elapsed, tput))
            except Exception as e:
                print(f"  {'GPU':<10}: SKIPPED — {e}")
                rows.append((n, 'GPU', None, None))
        print()

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    cols_per_scale = 5 if has_gpu else 4

    print("\n" + "=" * 80)
    print("MULTI-THREAD SCALING SUMMARY  (ingest + finalize only)")
    print("=" * 80)
    print(f"  {'Points':>12}  {'Mode':<10}  {'Time (s)':>9}  {'Mpt/s':>8}  {'vs CPU-1T':>10}")
    print("  " + "-" * 60)

    for i in range(0, len(rows), cols_per_scale):
        chunk = rows[i:i + cols_per_scale]
        baseline = chunk[0]   # CPU-1T is the first row
        n = baseline[0]
        for row in chunk:
            if row[2] is None:
                print(f"  {n:>12,}  {row[1]:<10}  {'SKIPPED':>9}")
                continue
            if row[1] == 'CPU-1T':
                speedup_str = "  <-- baseline"
            elif baseline[2] is not None:
                speedup = baseline[2] / row[2]
                speedup_str = f"  ({speedup:.2f}x)"
            else:
                speedup_str = ""
            print(f"  {n:>12,}  {row[1]:<10}  {row[2]:>9.3f}  {row[3]:>8.2f}{speedup_str}")
        print()

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    csv_path = "/workspace/scripts/multithread_benchmark_results.csv"
    with open(csv_path, 'w') as f:
        f.write("Points,Mode,Time_Seconds,Throughput_Mpts,Speedup_vs_CPU1T\n")
        for i in range(0, len(rows), cols_per_scale):
            chunk = rows[i:i + cols_per_scale]
            baseline_t = chunk[0][2]
            for row in chunk:
                if row[2] is None:
                    f.write(f"{row[0]},{row[1]},SKIPPED,,\n")
                else:
                    speedup = (baseline_t / row[2]) if baseline_t else 0.0
                    f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.2f},{speedup:.2f}\n")

    print(f"✓ Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
