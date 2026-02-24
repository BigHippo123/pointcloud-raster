#!/usr/bin/env python3
"""
Basic point cloud rasterization example using PCR Python API.

This example demonstrates:
1. Creating a synthetic point cloud
2. Configuring a rasterization grid
3. Running the pipeline
4. Saving the result as GeoTIFF
"""

import sys
import os
import numpy as np

# Add python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

import pcr


def create_synthetic_point_cloud(num_points=10000):
    """Create a synthetic point cloud with random points."""
    print(f"Creating synthetic point cloud with {num_points} points...")

    # Create point cloud with capacity
    cloud = pcr.PointCloud.create(num_points)

    # Add intensity channel
    cloud.add_channel("intensity", pcr.DataType.Float32)
    cloud.add_channel("class", pcr.DataType.Float32)

    # Generate random points in a 100x100 area
    np.random.seed(42)
    x = np.random.uniform(0, 100, num_points)
    y = np.random.uniform(0, 100, num_points)

    # Generate intensity values (higher near center)
    cx, cy = 50, 50
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    intensity = 100.0 * np.exp(-dist_from_center / 20.0)

    # Generate classification (2 = ground, 5 = vegetation)
    classification = np.where(intensity > 50, 2.0, 5.0)

    # Set data
    cloud.set_x_array(x)
    cloud.set_y_array(y)
    cloud.set_channel_array_f32("intensity", intensity.astype(np.float32))
    cloud.set_channel_array_f32("class", classification.astype(np.float32))

    # Set CRS
    cloud.set_crs(pcr.CRS.from_epsg(32633))  # UTM Zone 33N

    print(f"✓ Created point cloud: {cloud}")
    print(f"  Channels: {cloud.channel_names()}")
    print(f"  Intensity range: [{intensity.min():.2f}, {intensity.max():.2f}]")

    return cloud


def configure_pipeline(output_path="/tmp/pcr_example_output.tif"):
    """Configure the rasterization pipeline."""
    print("\nConfiguring pipeline...")

    # Create pipeline configuration
    config = pcr.PipelineConfig()

    # Configure output grid
    config.grid.bounds.min_x = 0.0
    config.grid.bounds.min_y = 0.0
    config.grid.bounds.max_x = 100.0
    config.grid.bounds.max_y = 100.0
    config.grid.cell_size_x = 1.0
    config.grid.cell_size_y = -1.0  # Negative for north-up convention
    config.grid.tile_width = 100  # Single tile for this small grid
    config.grid.tile_height = 100
    config.grid.crs = pcr.CRS.from_epsg(32633)  # UTM Zone 33N
    config.grid.compute_dimensions()

    print(f"✓ Grid configured: {config.grid.width}x{config.grid.height} cells")
    print(f"  Total cells: {config.grid.total_cells()}")
    print(f"  Tiles: {config.grid.tiles_x}x{config.grid.tiles_y}")

    # Configure reductions
    # 1. Average intensity
    reduction1 = pcr.ReductionSpec()
    reduction1.value_channel = "intensity"
    reduction1.type = pcr.ReductionType.Average
    reduction1.output_band_name = "mean_intensity"

    # 2. Maximum intensity
    reduction2 = pcr.ReductionSpec()
    reduction2.value_channel = "intensity"
    reduction2.type = pcr.ReductionType.Max
    reduction2.output_band_name = "max_intensity"

    # 3. Point count
    reduction3 = pcr.ReductionSpec()
    reduction3.value_channel = "intensity"
    reduction3.type = pcr.ReductionType.Count
    reduction3.output_band_name = "point_count"

    # Set reductions list
    config.reductions = [reduction1, reduction2, reduction3]

    print(f"✓ Reductions configured: {len(config.reductions)}")
    for i, r in enumerate(config.reductions):
        print(f"  {i+1}. {r.output_band_name} ({r.type})")

    # Optional: Add filter (only process ground points)
    # filter_spec = pcr.FilterSpec()
    # filter_spec.add("class", pcr.CompareOp.Equal, 2.0)
    # config.filter = filter_spec

    # Configure output
    config.output_path = output_path
    config.state_dir = "/tmp/pcr_example_tiles"
    config.write_cog = True  # Create Cloud-Optimized GeoTIFF

    # Memory budgets (0 = auto)
    config.host_cache_budget = 1024 * 1024 * 1024  # 1GB

    print(f"✓ Output: {config.output_path}")
    print(f"  State dir: {config.state_dir}")
    print(f"  COG: {config.write_cog}")

    return config


def run_pipeline(config, clouds):
    """Run the rasterization pipeline."""
    print("\nRunning pipeline...")

    # Create pipeline
    pipeline = pcr.Pipeline.create(config)
    if pipeline is None:
        raise RuntimeError("Failed to create pipeline")

    # Validate configuration
    pipeline.validate()
    print("✓ Pipeline validated")

    # Set up progress callback
    def progress_callback(info):
        print(f"  Progress: {info.points_processed} points, "
              f"{info.tiles_active} tiles active, "
              f"{info.elapsed_seconds:.2f}s elapsed")
        return True  # Return False to cancel

    pipeline.set_progress_callback(progress_callback)

    # Process point clouds
    for i, cloud in enumerate(clouds):
        print(f"  Ingesting cloud {i+1}/{len(clouds)}...")
        pipeline.ingest(cloud)

    # Finalize and write output
    print("  Finalizing...")
    pipeline.finalize()

    # Get statistics
    stats = pipeline.stats()
    print(f"✓ Pipeline complete!")
    print(f"  Total points processed: {stats.points_processed}")
    print(f"  Total time: {stats.elapsed_seconds:.2f}s")
    print(f"  Points/sec: {stats.points_processed / max(stats.elapsed_seconds, 0.001):.0f}")

    # Get result grid
    result = pipeline.result()
    if result:
        print(f"✓ Result grid: {result}")
        print(f"  Dimensions: {result.cols()}x{result.rows()}")
        print(f"  Bands: {result.num_bands()}")

        # Access band data as NumPy arrays
        for i in range(result.num_bands()):
            band_desc = result.band_desc(i)
            arr = result.band_array(i)
            valid_mask = ~np.isnan(arr)
            if valid_mask.any():
                print(f"  Band {i} ({band_desc.name}): "
                      f"min={np.nanmin(arr):.2f}, "
                      f"max={np.nanmax(arr):.2f}, "
                      f"mean={np.nanmean(arr):.2f}")
            else:
                print(f"  Band {i} ({band_desc.name}): all NaN")

    return pipeline


def main():
    """Main entry point."""
    print("=" * 70)
    print("PCR Python API - Basic Rasterization Example")
    print("=" * 70)

    # 1. Create synthetic point cloud
    cloud = create_synthetic_point_cloud(num_points=50000)

    # 2. Configure pipeline
    config = configure_pipeline(output_path="/tmp/pcr_python_example.tif")

    # 3. Run pipeline
    pipeline = run_pipeline(config, [cloud])

    print("\n" + "=" * 70)
    print("✅ Example complete!")
    print("=" * 70)
    print(f"\nOutput saved to: {config.output_path}")
    print("\nYou can view the output with:")
    print(f"  gdalinfo {config.output_path}")
    print(f"  gdal_translate -of PNG {config.output_path} /tmp/output.png")
    print()


if __name__ == "__main__":
    main()
