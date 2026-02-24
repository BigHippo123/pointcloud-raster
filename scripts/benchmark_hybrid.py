#!/usr/bin/env python3
"""
Benchmark Hybrid (CPU routing + GPU accumulation) vs pure CPU-MT vs pure GPU.

Shows whether the Hybrid mode beats pure GPU by overlapping CPU routing
preparation with GPU compute, and how it compares to CPU multi-threading.

Modes benchmarked for each scale:
  CPU-MT    : CPU mode, all cores (cpu_threads=0)
  GPU       : Pure GPU mode, cloud in host memory (pipeline does H2D)
  GPU-Device: Pure GPU mode, cloud pre-staged on GPU
  Hybrid-2T : Hybrid mode, 2 CPU routing threads
  Hybrid-NT : Hybrid mode, all available CPUs / 2 (hybrid_cpu_threads=0)

Point cloud sizes: 1M, 5M, 10M, 25M
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


def build_pipeline(mode, bbox, state_dir, output_path,
                   cpu_threads=0, hybrid_cpu_threads=0):
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
    cfg.hybrid_cpu_threads = hybrid_cpu_threads
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


def run_mode(label, mode, bbox, cloud, state_dir, output_path,
             cpu_threads=0, hybrid_cpu_threads=0):
    """Run one benchmark configuration, return (label, elapsed, throughput)."""
    try:
        pl = build_pipeline(mode, bbox, state_dir, output_path,
                            cpu_threads=cpu_threads,
                            hybrid_cpu_threads=hybrid_cpu_threads)
        elapsed = time_ingest_finalize(pl, cloud)
        tput = cloud.count() / elapsed / 1e6
        return (label, elapsed, tput)
    except Exception as e:
        print(f"    SKIPPED ({e})")
        return (label, None, None)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Hybrid vs CPU-MT vs GPU Performance Benchmark")
    print("=" * 80)

    gpu_util, gpu_mem = get_gpu_info()
    if gpu_util is None:
        print("\n✗ GPU not detected or nvidia-smi not available.")
        print("  Hybrid mode requires CUDA. Running CPU-only comparison instead.\n")
        has_gpu = False
    else:
        print(f"\n✓ GPU available  (utilization: {gpu_util}%,  memory used: {gpu_mem} MB)\n")
        has_gpu = True

    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 160.0
    bbox.max_y = 160.0

    scales = [1_000_000, 5_000_000, 10_000_000, 25_000_000]

    all_rows = []   # list of (n, label, elapsed, throughput)

    for n in scales:
        label_n = f"{n:,}"
        print(f"{'='*80}")
        print(f"Scale: {label_n} points")
        print(f"{'='*80}")

        host_cloud = make_random_cloud(n, bbox, on_device=False)
        scale_rows = []

        # -- CPU-MT --
        print(f"  CPU-MT       : ", end='', flush=True)
        row = run_mode('CPU-MT', pcr.ExecutionMode.CPU, bbox, host_cloud,
                       '/tmp/pcr_hybrid_cpu_mt', '/tmp/hybrid_cpu_mt.tif',
                       cpu_threads=0)
        elapsed_str = f"{row[1]:.3f}s" if row[1] else "SKIPPED"
        tput_str    = f"({row[2]:.2f} Mpt/s)" if row[2] else ""
        print(f"{elapsed_str}  {tput_str}")
        scale_rows.append((n,) + row)

        if has_gpu:
            # -- GPU (host cloud) --
            print(f"  GPU-Host     : ", end='', flush=True)
            row = run_mode('GPU-Host', pcr.ExecutionMode.GPU, bbox, host_cloud,
                           '/tmp/pcr_hybrid_gpu_h', '/tmp/hybrid_gpu_h.tif')
            elapsed_str = f"{row[1]:.3f}s" if row[1] else "SKIPPED"
            tput_str    = f"({row[2]:.2f} Mpt/s)" if row[2] else ""
            print(f"{elapsed_str}  {tput_str}")
            scale_rows.append((n,) + row)

            # -- GPU (device cloud) --
            print(f"  GPU-Device   : ", end='', flush=True)
            try:
                device_cloud = host_cloud.to_device()
                row = run_mode('GPU-Device', pcr.ExecutionMode.GPU, bbox, device_cloud,
                               '/tmp/pcr_hybrid_gpu_d', '/tmp/hybrid_gpu_d.tif')
                del device_cloud
                elapsed_str = f"{row[1]:.3f}s" if row[1] else "SKIPPED"
                tput_str    = f"({row[2]:.2f} Mpt/s)" if row[2] else ""
                print(f"{elapsed_str}  {tput_str}")
                scale_rows.append((n,) + row)
            except Exception as e:
                print(f"SKIPPED ({e})")
                scale_rows.append((n, 'GPU-Device', None, None))

            # -- Hybrid-2T --
            print(f"  Hybrid-2T    : ", end='', flush=True)
            row = run_mode('Hybrid-2T', pcr.ExecutionMode.Hybrid, bbox, host_cloud,
                           '/tmp/pcr_hybrid_h2t', '/tmp/hybrid_h2t.tif',
                           hybrid_cpu_threads=2)
            elapsed_str = f"{row[1]:.3f}s" if row[1] else "SKIPPED"
            tput_str    = f"({row[2]:.2f} Mpt/s)" if row[2] else ""
            print(f"{elapsed_str}  {tput_str}")
            scale_rows.append((n,) + row)

            # -- Hybrid-NT (all CPUs / 2) --
            print(f"  Hybrid-NT    : ", end='', flush=True)
            row = run_mode('Hybrid-NT', pcr.ExecutionMode.Hybrid, bbox, host_cloud,
                           '/tmp/pcr_hybrid_hnt', '/tmp/hybrid_hnt.tif',
                           hybrid_cpu_threads=0)
            elapsed_str = f"{row[1]:.3f}s" if row[1] else "SKIPPED"
            tput_str    = f"({row[2]:.2f} Mpt/s)" if row[2] else ""
            print(f"{elapsed_str}  {tput_str}")
            scale_rows.append((n,) + row)

        all_rows.extend(scale_rows)
        print()

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    cols = 6 if has_gpu else 1   # CPU-MT + GPU-Host + GPU-Device + Hybrid-2T + Hybrid-NT

    print("\n" + "=" * 80)
    print("HYBRID BENCHMARK SUMMARY  (ingest + finalize only)")
    print("=" * 80)
    print(f"  {'Points':>12}  {'Mode':<12}  {'Time (s)':>9}  {'Mpt/s':>8}  {'vs CPU-MT':>10}")
    print("  " + "-" * 65)

    rows_per_scale = 1 + (4 if has_gpu else 0)  # CPU-MT + GPU modes + Hybrid modes

    for i in range(0, len(all_rows), rows_per_scale):
        chunk = all_rows[i:i + rows_per_scale]
        baseline = chunk[0]   # CPU-MT is baseline
        n = baseline[0]
        for row in chunk:
            if row[2] is None:
                print(f"  {n:>12,}  {row[1]:<12}  {'SKIPPED':>9}")
                continue
            if row[1] == 'CPU-MT':
                speedup_str = "  <-- baseline"
            elif baseline[2] is not None:
                speedup = baseline[2] / row[2]
                speedup_str = f"  ({speedup:.2f}x)"
            else:
                speedup_str = ""
            print(f"  {n:>12,}  {row[1]:<12}  {row[2]:>9.3f}  {row[3]:>8.2f}{speedup_str}")
        print()

    # -------------------------------------------------------------------------
    # Save CSV
    # -------------------------------------------------------------------------
    csv_path = "/workspace/scripts/hybrid_benchmark_results.csv"
    with open(csv_path, 'w') as f:
        f.write("Points,Mode,Time_Seconds,Throughput_Mpts,Speedup_vs_CPU_MT\n")
        for i in range(0, len(all_rows), rows_per_scale):
            chunk = all_rows[i:i + rows_per_scale]
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
