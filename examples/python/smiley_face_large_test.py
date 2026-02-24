#!/usr/bin/env python3
"""
Large Smiley Face Test - scaled down version to verify the approach

Uses:
- 10,000 x 10,000 grid (100M cells)
- 100,000,000 points (100 million) streamed in chunks
- Memory tracking

This is a test run before the full 1M x 1M grid with 10B points.
"""

import sys
import os
import numpy as np
import time
import psutil
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))
import pcr


class MemoryTracker:
    def __init__(self):
        self.process = psutil.Process()
        self.peak_rss = 0
        self.measurements = []
        self.start_time = time.time()

    def measure(self, label=""):
        gc.collect()
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024
        vms_mb = mem_info.vms / 1024 / 1024
        self.peak_rss = max(self.peak_rss, rss_mb)
        elapsed = time.time() - self.start_time

        measurement = {'time': elapsed, 'label': label, 'rss_mb': rss_mb, 'vms_mb': vms_mb}
        self.measurements.append(measurement)

        print(f"[{elapsed:7.2f}s] {label:30s} RSS: {rss_mb:8.1f} MB")
        return measurement

    def report(self):
        print("\n" + "=" * 70)
        print("PERFORMANCE REPORT")
        print("=" * 70)
        print(f"Peak RSS:     {self.peak_rss:10.1f} MB")
        print(f"Total time:   {time.time() - self.start_time:10.2f} seconds")
        print("=" * 70)


def generate_batch(batch_num, points_per_batch, scale=100.0):
    """Generate one batch of smiley points."""
    # Smiley centered at (5000, 5000) on 10k x 10k grid
    cx, cy = 5000.0, 5000.0

    # Features
    n_eyes = int(points_per_batch * 0.10)
    n_smile = int(points_per_batch * 0.05)
    n_bg = points_per_batch - n_eyes - n_smile

    all_x, all_y, all_i = [], [], []

    # Eyes
    if n_eyes > 0:
        n_per_eye = n_eyes // 2
        # Left eye
        angles = np.random.uniform(0, 2 * np.pi, n_per_eye)
        radii = np.random.uniform(0, 500, n_per_eye)
        all_x.append(cx - 1500 + radii * np.cos(angles))
        all_y.append(cy + 1000 + radii * np.sin(angles))
        all_i.append(np.full(n_per_eye, 200.0, dtype=np.float32))

        # Right eye
        all_x.append(cx + 1500 + radii * np.cos(angles))
        all_y.append(cy + 1000 + radii * np.sin(angles))
        all_i.append(np.full(n_per_eye, 200.0, dtype=np.float32))

    # Smile
    if n_smile > 0:
        angles = np.random.uniform(-2.8, -0.34, n_smile)
        radii = 2000 + np.random.uniform(-50, 50, n_smile)
        all_x.append(cx + radii * np.cos(angles))
        all_y.append(cy - 500 + radii * np.sin(angles))
        all_i.append(np.full(n_smile, 180.0, dtype=np.float32))

    # Background
    if n_bg > 0:
        all_x.append(np.random.uniform(500, 9500, n_bg))
        all_y.append(np.random.uniform(500, 9500, n_bg))
        all_i.append(np.random.uniform(5, 30, n_bg).astype(np.float32))

    return np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_i)


def main():
    mem = MemoryTracker()
    mem.measure("Start")

    print("=" * 70)
    print("LARGE SMILEY TEST (scaled down)")
    print("=" * 70)
    print()
    print("  Grid:         10,000 x 10,000 cells")
    print("  Points:       100,000,000 (100 million)")
    print("  Batch size:   1,000,000 points")
    print("  Batches:      100")
    print()

    # Configure
    config = pcr.PipelineConfig()
    config.grid.bounds.min_x = 0.0
    config.grid.bounds.min_y = 0.0
    config.grid.bounds.max_x = 10000.0
    config.grid.bounds.max_y = 10000.0
    config.grid.cell_size_x = 1.0
    config.grid.cell_size_y = -1.0
    config.grid.tile_width = 1024
    config.grid.tile_height = 1024
    config.grid.crs = pcr.CRS.from_epsg(32633)
    config.grid.compute_dimensions()

    reduction = pcr.ReductionSpec()
    reduction.value_channel = "intensity"
    reduction.type = pcr.ReductionType.Average
    config.reductions = [reduction]

    config.output_path = "/tmp/large_smiley_test.tif"
    config.state_dir = "/tmp/large_smiley_tiles"
    config.write_cog = True
    config.host_cache_budget = 2 * 1024 * 1024 * 1024  # 2GB

    print(f"Grid: {config.grid.width} x {config.grid.height} = {config.grid.total_cells():,} cells")
    print(f"Tiles: {config.grid.tiles_x} x {config.grid.tiles_y} = {config.grid.total_tiles()} tiles")
    print()

    mem.measure("Config created")

    # Create pipeline
    pipeline = pcr.Pipeline.create(config)
    pipeline.validate()
    mem.measure("Pipeline created")

    # Process batches
    total_points = 100_000_000
    points_per_batch = 1_000_000
    total_batches = total_points // points_per_batch

    print("Processing batches...")
    for batch in range(total_batches):
        x, y, intensity = generate_batch(batch, points_per_batch)

        cloud = pcr.PointCloud.create(points_per_batch)
        cloud.add_channel("intensity", pcr.DataType.Float32)
        cloud.set_x_array(x)
        cloud.set_y_array(y)
        cloud.set_channel_array_f32("intensity", intensity)
        cloud.set_crs(config.grid.crs)

        pipeline.ingest(cloud)

        del cloud, x, y, intensity

        if (batch + 1) % 10 == 0:
            mem.measure(f"Batch {batch + 1}/{total_batches}")

    print()
    mem.measure("All batches ingested")

    # Finalize
    print("Finalizing...")
    pipeline.finalize()
    mem.measure("Finalized")

    stats = pipeline.stats()
    print()
    print(f"Processed: {stats.points_processed:,} points in {stats.elapsed_seconds:.2f}s")
    print(f"Rate: {stats.points_processed / stats.elapsed_seconds:,.0f} points/sec")

    mem.report()

    print()
    print(f"Output: {config.output_path}")
    print("✓ Test complete!")


if __name__ == "__main__":
    main()
