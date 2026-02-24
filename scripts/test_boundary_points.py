#!/usr/bin/env python3
"""
Test what happens with boundary points specifically.
"""

import os
import sys
import shutil
import numpy as np
sys.path.insert(0, '/workspace/python')
import pcr

def test_explicit_boundary_points():
    """Test with explicitly crafted boundary points."""
    print("Testing with explicit boundary points at grid edges...")

    # Create 160x160 grid
    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 160.0
    bbox.max_y = 160.0

    grid_config = pcr.GridConfig()
    grid_config.bounds = bbox
    grid_config.crs = pcr.CRS.from_epsg(32610)
    grid_config.cell_size_x = 1.0
    grid_config.cell_size_y = -1.0
    grid_config.compute_dimensions()

    print(f"Grid: {grid_config.width} × {grid_config.height}")

    # Create test points at various Y coordinates
    test_y_coords = [
        0.0,          # Exactly at min_y (bottom)
        0.000001,     # Just above min_y
        0.5,          # Mid-cell at bottom
        159.5,        # Mid-cell near top
        159.999999,   # Just below max_y
        160.0,        # Exactly at max_y (top)
    ]

    print(f"\nTesting Y coordinates:")
    for y in test_y_coords:
        x = 80.0  # Middle of grid in X

        # Manual calculation of what cell index should be
        origin_y = bbox.max_y  # 160.0
        row_calc = (y - origin_y) / grid_config.cell_size_y
        row_int = int(np.floor(row_calc))

        print(f"  y={y:10.6f}: row_calc={row_calc:10.6f}, row_int={row_int:3d}", end="")

        if row_int < 0 or row_int >= grid_config.height:
            print(f" → OUT OF BOUNDS")
        else:
            print(f" → valid")

    # Now test with actual pipeline
    print(f"\n{'='*70}")
    print("Testing with safe interior points (epsilon=1.0)...")
    print(f"{'='*70}")

    # Create point cloud with safe interior points
    epsilon = 1.0  # Stay 1 meter inside boundaries
    num_points = 100

    points_x = np.random.uniform(bbox.min_x + epsilon, bbox.max_x - epsilon, num_points)
    points_y = np.random.uniform(bbox.min_y + epsilon, bbox.max_y - epsilon, num_points)
    values = np.full(num_points, 50.0, dtype=np.float32)

    print(f"Point range: X=[{points_x.min():.6f}, {points_x.max():.6f}], "
          f"Y=[{points_y.min():.6f}, {points_y.max():.6f}]")

    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x.astype(np.float64))
    cloud.set_y_array(points_y.astype(np.float64))
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    # Configure pipeline - EXPLICITLY FORCE CPU MODE
    state_dir = "/tmp/pcr_boundary_test"
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.makedirs(state_dir, exist_ok=True)

    reduction = pcr.ReductionSpec()
    reduction.value_channel = "value"
    reduction.type = pcr.ReductionType.Average
    reduction.output_band_name = "test"

    config = pcr.PipelineConfig()
    config.grid = grid_config
    config.reductions = [reduction]
    config.output_path = "/tmp/test_boundary.tif"
    config.state_dir = state_dir

    # CRITICAL: Explicitly set CPU mode
    config.exec_mode = pcr.ExecutionMode.CPU

    print(f"Pipeline config: exec_mode = CPU")

    try:
        pipeline = pcr.Pipeline.create(config)
        print(f"Pipeline created")

        pipeline.ingest(cloud)
        print(f"✓ Ingest successful!")

        pipeline.finalize()
        print(f"✓ Finalize successful!")

        stats = pipeline.stats()
        print(f"✓ Processed {stats.points_processed} points")

    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False

    return True

if __name__ == "__main__":
    test_explicit_boundary_points()
