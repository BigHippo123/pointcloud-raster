#!/usr/bin/env python3
"""
Benchmark CPU vs GPU pipeline performance across multiple point-cloud sizes.

Measures ingest+finalize throughput only (Pipeline.create is excluded from
timing so one-time GPU init cost does not distort per-run results).

Two GPU paths are tested for each size:
  GPU-Host  : cloud stays in host memory; pipeline does the H2D transfer.
  GPU-Device: cloud is pre-staged on the GPU before timing starts, removing
              H2D transfer cost from the measurement.
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

def get_gpu_info():
    """Return (util_pct, mem_used_mb) from nvidia-smi, or (None, None)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            gpu_util, mem_used = result.stdout.strip().split(',')
            return int(gpu_util.strip()), int(mem_used.strip())
    except Exception:
        pass
    return None, None


def make_random_cloud(n, bbox, on_device=False):
    """Create a random point cloud with a 'value' channel."""
    cloud = pcr.PointCloud.create(n)
    rng = np.random.default_rng(42)
    xs = rng.uniform(bbox.min_x + 0.5, bbox.max_x - 0.5, n).astype(np.float64)
    ys = rng.uniform(bbox.min_y + 0.5, bbox.max_y - 0.5, n).astype(np.float64)
    vs = rng.uniform(0, 100, n).astype(np.float32)
    cloud.set_x_array(xs)
    cloud.set_y_array(ys)
    cloud.add_channel('value', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value', vs)
    if on_device:
        cloud = cloud.to_device()
    return cloud


def build_pipeline(mode, bbox, state_dir, output_path, cpu_threads=0):
    """Create and return a Pipeline (not timed)."""
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
    """Return wall-clock seconds for ingest + finalize."""
    t0 = time.perf_counter()
    pipeline.ingest(cloud)
    pipeline.finalize()
    return time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("CPU vs GPU Performance Benchmark")
    print("=" * 80)

    gpu_util, gpu_mem = get_gpu_info()
    if gpu_util is None:
        print("\n✗ GPU not detected or nvidia-smi not available")
        return
    print(f"\n✓ GPU available  (utilization: {gpu_util}%,  memory used: {gpu_mem} MB)\n")

    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 160.0
    bbox.max_y = 160.0

    scales = [1_000_000, 5_000_000, 10_000_000, 25_000_000]

    rows = []   # (n, label, elapsed, throughput_mpts)

    for n in scales:
        label_n = f"{n:,}"
        print(f"{'='*80}")
        print(f"Scale: {label_n} points")
        print(f"{'='*80}")

        # -- CPU (single-thread baseline) --
        host_cloud = make_random_cloud(n, bbox, on_device=False)
        pl = build_pipeline(pcr.ExecutionMode.CPU, bbox,
                            '/tmp/pcr_bench_cpu', '/tmp/bench_cpu.tif',
                            cpu_threads=1)
        elapsed = time_ingest_finalize(pl, host_cloud)
        tput = n / elapsed / 1e6
        print(f"  CPU (1T)     : {elapsed:.3f}s  ({tput:.2f} Mpt/s)")
        rows.append((n, 'CPU', elapsed, tput))

        # -- CPU-MT (all cores) --
        pl = build_pipeline(pcr.ExecutionMode.CPU, bbox,
                            '/tmp/pcr_bench_cpu_mt', '/tmp/bench_cpu_mt.tif',
                            cpu_threads=0)
        elapsed = time_ingest_finalize(pl, host_cloud)
        tput = n / elapsed / 1e6
        print(f"  CPU (MT)     : {elapsed:.3f}s  ({tput:.2f} Mpt/s)")
        rows.append((n, 'CPU-MT', elapsed, tput))

        # -- GPU (host cloud — pipeline does H2D) --
        pl = build_pipeline(pcr.ExecutionMode.GPU, bbox,
                            '/tmp/pcr_bench_gpu_h', '/tmp/bench_gpu_h.tif')
        elapsed = time_ingest_finalize(pl, host_cloud)
        tput = n / elapsed / 1e6
        print(f"  GPU (host)   : {elapsed:.3f}s  ({tput:.2f} Mpt/s)")
        rows.append((n, 'GPU-Host', elapsed, tput))

        # -- GPU (device cloud — H2D already done before timing) --
        # Free the previous GPU pipeline before allocating a device cloud to
        # avoid OOM from holding both simultaneously on large point counts.
        del pl
        try:
            device_cloud = host_cloud.to_device()
            pl2 = build_pipeline(pcr.ExecutionMode.GPU, bbox,
                                '/tmp/pcr_bench_gpu_d', '/tmp/bench_gpu_d.tif')
            elapsed = time_ingest_finalize(pl2, device_cloud)
            tput = n / elapsed / 1e6
            gpu_u, _ = get_gpu_info()
            print(f"  GPU (device) : {elapsed:.3f}s  ({tput:.2f} Mpt/s)"
                  + (f"  [util {gpu_u}%]" if gpu_u is not None else ""))
            rows.append((n, 'GPU-Device', elapsed, tput))
            del pl2, device_cloud
        except RuntimeError as e:
            print(f"  GPU (device) : SKIPPED — {e}")
            rows.append((n, 'GPU-Device', None, None))
        print()

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY  (ingest + finalize, Pipeline.create excluded)")
    print("=" * 80)
    print(f"  {'Points':>12}  {'Mode':<12}  {'Time (s)':>9}  {'Mpt/s':>8}  {'Speedup':>9}")
    print("  " + "-" * 60)

    # 4 rows per scale: CPU(1T), CPU-MT, GPU-Host, GPU-Device
    for i in range(0, len(rows), 4):
        chunk = rows[i:i + 4]
        cpu_row = chunk[0]   # CPU single-thread is the baseline
        n = cpu_row[0]
        for row in chunk:
            if row[2] is None:
                print(f"  {n:>12,}  {row[1]:<12}  {'SKIPPED':>9}  {'':>8}")
                continue
            speedup = cpu_row[2] / row[2]
            marker = "  <-- baseline" if row[1] == 'CPU' else f"  ({speedup:.1f}x vs CPU-1T)"
            print(f"  {n:>12,}  {row[1]:<12}  {row[2]:>9.3f}  {row[3]:>8.2f}{marker}")
        print()

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    csv_path = "/workspace/scripts/performance_comparison.csv"
    with open(csv_path, 'w') as f:
        f.write("Points,Mode,Time_Seconds,Throughput_Mpts,Speedup_vs_CPU\n")
        for i in range(0, len(rows), 4):
            chunk = rows[i:i + 4]
            cpu_t = chunk[0][2]
            for row in chunk:
                if row[2] is None:
                    f.write(f"{row[0]},{row[1]},SKIPPED,,\n")
                else:
                    speedup = (cpu_t / row[2]) if cpu_t else 0.0
                    f.write(f"{row[0]},{row[1]},{row[2]:.4f},{row[3]:.2f},{speedup:.2f}\n")
    print(f"✓ Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
