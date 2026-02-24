#!/usr/bin/env python3
"""
Benchmark: Point vs Line vs Gaussian glyph rasterization.

Measures ingest+finalize throughput for different glyph types across
increasing point counts and sigma values.

Usage:
    python3 scripts/benchmark_glyph.py [--n 1000000] [--mode cpu|gpu|both]
"""

import os
import sys
import time
import argparse
import tempfile
import shutil

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import pcr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bbox(x0, y0, x1, y1):
    b = pcr.BBox()
    b.min_x, b.min_y, b.max_x, b.max_y = x0, y0, x1, y1
    return b


def make_cloud(n, bbox, rng, add_glyph_channels=True):
    cloud = pcr.PointCloud.create(n)
    xs = rng.uniform(bbox.min_x + 1, bbox.max_x - 1, n)
    ys = rng.uniform(bbox.min_y + 1, bbox.max_y - 1, n)
    vs = rng.uniform(0, 1, n).astype(np.float32)
    cloud.set_x_array(xs)
    cloud.set_y_array(ys)
    cloud.add_channel('value', pcr.DataType.Float32)
    cloud.set_channel_array_f32('value', vs)
    if add_glyph_channels:
        cloud.add_channel('sigma', pcr.DataType.Float32)
        cloud.add_channel('direction', pcr.DataType.Float32)
        cloud.add_channel('half_length', pcr.DataType.Float32)
        cloud.set_channel_array_f32('sigma', np.full(n, 2.0, np.float32))
        cloud.set_channel_array_f32('direction', rng.uniform(0, np.pi, n).astype(np.float32))
        cloud.set_channel_array_f32('half_length', np.full(n, 2.0, np.float32))
    return cloud


def run_benchmark(spec, cloud, gc, mode, tmpdir_base):
    tmpdir = tempfile.mkdtemp(dir=tmpdir_base)
    cfg = pcr.PipelineConfig()
    cfg.grid = gc
    cfg.reductions = [spec]
    cfg.exec_mode = mode
    cfg.gpu_fallback_to_cpu = True
    cfg.state_dir = os.path.join(tmpdir, 'state')
    cfg.output_path = os.path.join(tmpdir, 'out.tif')

    pipe = pcr.Pipeline.create(cfg)
    if pipe is None:
        shutil.rmtree(tmpdir)
        return None

    t0 = time.perf_counter()
    pipe.ingest(cloud)
    pipe.finalize()
    elapsed = time.perf_counter() - t0

    shutil.rmtree(tmpdir)
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Benchmark glyph rasterization')
    parser.add_argument('--mode', choices=['cpu', 'gpu', 'both'], default='both')
    parser.add_argument('--sizes', nargs='+', type=int,
                        default=[100_000, 1_000_000, 5_000_000])
    parser.add_argument('--sigmas', nargs='+', type=float, default=[1.0, 4.0, 16.0])
    args = parser.parse_args()

    modes = []
    if args.mode in ('cpu', 'both'):
        modes.append(('CPU', pcr.ExecutionMode.CPU))
    if args.mode in ('gpu', 'both'):
        modes.append(('GPU', pcr.ExecutionMode.GPU))

    rng = np.random.default_rng(42)
    bbox = make_bbox(0, 0, 1000, 1000)
    gc = pcr.GridConfig()
    gc.bounds = bbox
    gc.cell_size_x = 1.0
    gc.cell_size_y = -1.0
    gc.compute_dimensions()

    tmpdir_base = tempfile.mkdtemp(prefix='pcr_bench_glyph_')

    print(f"\nGrid: {gc.width} x {gc.height} = {gc.width * gc.height:,} cells")
    print(f"{'Mode':<6} {'Glyph':<12} {'N':>10} {'Sigma':>8} {'Time(s)':>9} {'Mpts/s':>8}")
    print('-' * 62)

    try:
        for n in args.sizes:
            cloud = make_cloud(n, bbox, rng)

            for mode_name, mode in modes:
                # --- Point glyph (baseline) ---
                spec_pt = pcr.ReductionSpec()
                spec_pt.value_channel = 'value'
                spec_pt.type = pcr.ReductionType.Average
                spec_pt.output_band_name = 'out'
                t = run_benchmark(spec_pt, cloud, gc, mode, tmpdir_base)
                if t is not None:
                    print(f"{mode_name:<6} {'Point':<12} {n:>10,} {'N/A':>8} {t:>9.3f} {n/t/1e6:>8.2f}")

                # --- Line glyph ---
                spec_ln = pcr.line_splat_spec('value',
                    direction_channel='direction',
                    half_length_channel='half_length',
                    max_radius_cells=8.0)
                t = run_benchmark(spec_ln, cloud, gc, mode, tmpdir_base)
                if t is not None:
                    print(f"{mode_name:<6} {'Line':<12} {n:>10,} {'N/A':>8} {t:>9.3f} {n/t/1e6:>8.2f}")

                # --- Gaussian glyph at different sigmas ---
                for sigma in args.sigmas:
                    spec_gs = pcr.gaussian_splat_spec('value',
                        default_sigma=sigma,
                        max_radius_cells=min(4 * sigma, 64.0))
                    t = run_benchmark(spec_gs, cloud, gc, mode, tmpdir_base)
                    if t is not None:
                        print(f"{mode_name:<6} {'Gaussian':<12} {n:>10,} {sigma:>8.1f} {t:>9.3f} {n/t/1e6:>8.2f}")

            print()

    finally:
        shutil.rmtree(tmpdir_base, ignore_errors=True)


if __name__ == '__main__':
    main()
