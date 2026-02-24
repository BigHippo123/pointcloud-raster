#!/usr/bin/env python3
"""
MASSIVE Smiley Face Point Cloud Rasterization Demo

Demonstrates out-of-core processing with:
- 1,000,000 x 1,000,000 grid (1 trillion cells!)
- 10 billion points (streamed in chunks)
- Memory usage tracking
- Performance logging

This showcases PCR's ability to handle datasets that far exceed available RAM.
"""

import sys
import os
import numpy as np
import time
import psutil
import gc

# Add python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

import pcr


class MemoryTracker:
    """Track memory usage throughout processing."""

    def __init__(self):
        self.process = psutil.Process()
        self.peak_rss = 0
        self.measurements = []
        self.start_time = time.time()

    def measure(self, label=""):
        """Take a memory measurement."""
        gc.collect()  # Force garbage collection for accurate measurement
        mem_info = self.process.memory_info()
        rss_mb = mem_info.rss / 1024 / 1024
        vms_mb = mem_info.vms / 1024 / 1024

        self.peak_rss = max(self.peak_rss, rss_mb)

        elapsed = time.time() - self.start_time

        measurement = {
            'time': elapsed,
            'label': label,
            'rss_mb': rss_mb,
            'vms_mb': vms_mb
        }
        self.measurements.append(measurement)

        print(f"[{elapsed:7.2f}s] {label:30s} RSS: {rss_mb:8.1f} MB  VMS: {vms_mb:8.1f} MB")

        return measurement

    def report(self):
        """Print final memory report."""
        print("\n" + "=" * 80)
        print("MEMORY & PERFORMANCE REPORT")
        print("=" * 80)
        print(f"Peak RSS:        {self.peak_rss:10.1f} MB")
        print(f"Total time:      {time.time() - self.start_time:10.2f} seconds")
        print(f"Measurements:    {len(self.measurements)}")
        print()

        if len(self.measurements) > 1:
            print("Memory Timeline:")
            print(f"{'Time (s)':>10} {'Label':30} {'RSS (MB)':>12} {'Delta (MB)':>12}")
            print("-" * 80)

            prev_rss = self.measurements[0]['rss_mb']
            for m in self.measurements:
                delta = m['rss_mb'] - prev_rss
                print(f"{m['time']:10.2f} {m['label']:30} {m['rss_mb']:12.1f} {delta:+12.1f}")
                prev_rss = m['rss_mb']

        print("=" * 80)


def generate_smiley_points_batch(batch_num, points_per_batch, total_batches):
    """Generate one batch of points for the smiley face."""

    # Scale factor for 1M x 1M grid (smiley centered at 500k, 500k)
    scale = 10000.0  # 1 meter = 10000 grid units

    # Face dimensions (in grid coordinates)
    face_center_x = 500000.0
    face_center_y = 500000.0

    # Features at appropriate scale
    left_eye_x = face_center_x - 150000.0
    left_eye_y = face_center_y + 100000.0
    left_eye_radius = 50000.0

    right_eye_x = face_center_x + 150000.0
    right_eye_y = face_center_y + 100000.0
    right_eye_radius = 50000.0

    smile_center_y = face_center_y - 50000.0
    smile_radius = 200000.0
    smile_start_angle = -2.8
    smile_end_angle = -0.34

    # Determine what to generate this batch
    # Distribute points: 10% eyes, 5% smile, 85% background
    points_in_eyes = int(points_per_batch * 0.10)
    points_in_smile = int(points_per_batch * 0.05)
    points_in_background = points_per_batch - points_in_eyes - points_in_smile

    all_x = []
    all_y = []
    all_intensity = []

    # Left eye (circle)
    if points_in_eyes > 0:
        n_per_eye = points_in_eyes // 2
        angles = np.random.uniform(0, 2 * np.pi, n_per_eye)
        radii = np.random.uniform(0, left_eye_radius, n_per_eye)
        x = left_eye_x + radii * np.cos(angles)
        y = left_eye_y + radii * np.sin(angles)
        intensity = np.full(n_per_eye, 200.0, dtype=np.float32)

        all_x.append(x)
        all_y.append(y)
        all_intensity.append(intensity)

        # Right eye
        x = right_eye_x + radii * np.cos(angles)
        y = right_eye_y + radii * np.sin(angles)
        all_x.append(x)
        all_y.append(y)
        all_intensity.append(intensity)

    # Smile (arc)
    if points_in_smile > 0:
        angles = np.random.uniform(smile_start_angle, smile_end_angle, points_in_smile)
        radii = smile_radius + np.random.uniform(-5000, 5000, points_in_smile)
        x = face_center_x + radii * np.cos(angles)
        y = smile_center_y + radii * np.sin(angles)
        intensity = np.full(points_in_smile, 180.0, dtype=np.float32)

        all_x.append(x)
        all_y.append(y)
        all_intensity.append(intensity)

    # Background (random across entire grid)
    if points_in_background > 0:
        x = np.random.uniform(50000, 950000, points_in_background)
        y = np.random.uniform(50000, 950000, points_in_background)
        intensity = np.random.uniform(5, 30, points_in_background).astype(np.float32)

        all_x.append(x)
        all_y.append(y)
        all_intensity.append(intensity)

    # Combine all points
    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    intensity_all = np.concatenate(all_intensity)

    return x_all, y_all, intensity_all


def create_massive_smiley_pipeline():
    """Configure the pipeline for massive processing."""

    print("=" * 80)
    print("🙂 MASSIVE SMILEY FACE RASTERIZATION 🙂")
    print("=" * 80)
    print()
    print("Configuration:")
    print(f"  Grid:           1,000,000 x 1,000,000 cells (1 trillion cells)")
    print(f"  Resolution:     1.0 meter per cell")
    print(f"  Total points:   10,000,000,000 (10 billion)")
    print(f"  Chunk size:     10,000,000 points per batch (1000 batches)")
    print(f"  Tile size:      4096 x 4096 cells")
    print(f"  Memory budget:  4 GB cache")
    print()

    # Create pipeline configuration
    config = pcr.PipelineConfig()

    # Grid: 1M x 1M cells, 1m resolution
    config.grid.bounds.min_x = 0.0
    config.grid.bounds.min_y = 0.0
    config.grid.bounds.max_x = 1000000.0
    config.grid.bounds.max_y = 1000000.0
    config.grid.cell_size_x = 1.0
    config.grid.cell_size_y = -1.0
    config.grid.tile_width = 4096  # Standard tile size
    config.grid.tile_height = 4096
    config.grid.crs = pcr.CRS.from_epsg(32633)
    config.grid.compute_dimensions()

    print(f"Grid computed:")
    print(f"  Dimensions:     {config.grid.width} x {config.grid.height}")
    print(f"  Total cells:    {config.grid.total_cells():,}")
    print(f"  Tiles:          {config.grid.tiles_x} x {config.grid.tiles_y} = {config.grid.total_tiles():,} tiles")
    print()

    # Reduction: Average intensity
    reduction = pcr.ReductionSpec()
    reduction.value_channel = "intensity"
    reduction.type = pcr.ReductionType.Average
    reduction.output_band_name = "mean_intensity"
    config.reductions = [reduction]

    # Output configuration
    output_path = "/tmp/massive_smiley_face.tif"
    config.output_path = output_path
    config.state_dir = "/tmp/massive_smiley_tiles"
    config.write_cog = True

    # Memory budget (4GB for tile cache)
    config.host_cache_budget = 4 * 1024 * 1024 * 1024

    print(f"Output:")
    print(f"  Path:           {output_path}")
    print(f"  State dir:      {config.state_dir}")
    print(f"  COG:            {config.write_cog}")
    print(f"  Cache budget:   {config.host_cache_budget / 1024 / 1024 / 1024:.1f} GB")
    print()

    return config


def process_massive_smiley():
    """Main processing function."""

    mem_tracker = MemoryTracker()
    mem_tracker.measure("Initial")

    # Configure pipeline
    config = create_massive_smiley_pipeline()
    mem_tracker.measure("Pipeline configured")

    # Create pipeline
    print("Creating pipeline...")
    pipeline = pcr.Pipeline.create(config)
    if not pipeline:
        raise RuntimeError("Failed to create pipeline")

    pipeline.validate()
    print("✓ Pipeline validated")
    mem_tracker.measure("Pipeline created")

    # Processing parameters
    total_points = 10_000_000_000  # 10 billion
    points_per_batch = 10_000_000   # 10 million per batch
    total_batches = total_points // points_per_batch

    print()
    print("=" * 80)
    print("STREAMING POINT GENERATION & INGESTION")
    print("=" * 80)
    print()

    # Progress tracking
    points_processed = 0
    batch_times = []

    # Set up progress callback
    last_progress_time = [time.time()]
    last_progress_points = [0]

    def progress_callback(info):
        now = time.time()
        if now - last_progress_time[0] >= 5.0:  # Report every 5 seconds
            elapsed = now - last_progress_time[0]
            points_delta = info.points_processed - last_progress_points[0]
            rate = points_delta / elapsed if elapsed > 0 else 0

            print(f"  Progress: {info.points_processed:,} points "
                  f"({info.points_processed / total_points * 100:.1f}%), "
                  f"{info.tiles_active} tiles active, "
                  f"{rate:,.0f} pts/sec")

            last_progress_time[0] = now
            last_progress_points[0] = info.points_processed

        return True  # Continue processing

    pipeline.set_progress_callback(progress_callback)

    # Process in batches
    for batch_num in range(total_batches):
        batch_start = time.time()

        # Generate point batch
        x, y, intensity = generate_smiley_points_batch(
            batch_num, points_per_batch, total_batches
        )

        # Create point cloud for this batch
        cloud = pcr.PointCloud.create(points_per_batch)
        cloud.add_channel("intensity", pcr.DataType.Float32)
        cloud.set_x_array(x)
        cloud.set_y_array(y)
        cloud.set_channel_array_f32("intensity", intensity)
        cloud.set_crs(pcr.CRS.from_epsg(32633))

        # Ingest batch
        pipeline.ingest(cloud)

        points_processed += points_per_batch
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        # Clean up to free memory
        del cloud, x, y, intensity

        # Report progress every 100 batches
        if (batch_num + 1) % 100 == 0:
            avg_batch_time = np.mean(batch_times[-100:])
            rate = points_per_batch / avg_batch_time
            mem_tracker.measure(f"Batch {batch_num + 1}/{total_batches}")
            print(f"Batch {batch_num + 1}/{total_batches}: "
                  f"{points_processed:,} / {total_points:,} points "
                  f"({batch_time:.2f}s, {rate:,.0f} pts/sec avg)")

    print()
    print(f"✓ All {total_points:,} points generated and ingested")
    mem_tracker.measure("All points ingested")

    # Finalize
    print()
    print("=" * 80)
    print("FINALIZING RESULT")
    print("=" * 80)
    print()
    print("Flushing tiles and assembling final grid...")
    print("(This may take several minutes for large grids)")
    print()

    finalize_start = time.time()
    pipeline.finalize()
    finalize_time = time.time() - finalize_start

    print(f"✓ Finalization complete in {finalize_time:.1f} seconds")
    mem_tracker.measure("Result finalized")

    # Get statistics
    stats = pipeline.stats()

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Total points processed:  {stats.points_processed:,}")
    print(f"Total time:              {stats.elapsed_seconds:.2f} seconds")
    print(f"Average rate:            {stats.points_processed / stats.elapsed_seconds:,.0f} points/sec")
    print(f"Tiles active:            {stats.tiles_active}")
    print()

    # Memory report
    mem_tracker.report()

    print()
    print("Output saved to:", config.output_path)
    print()
    print("View with:")
    print(f"  gdalinfo {config.output_path}")
    print(f"  gdal_translate -outsize 1000 1000 {config.output_path} /tmp/smiley_preview.tif")
    print(f"  gdal_translate -of PNG -scale /tmp/smiley_preview.tif /tmp/smiley_preview.png")
    print()
    print("🙂 Massive processing complete! 🙂")
    print()


if __name__ == "__main__":
    try:
        process_massive_smiley()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
