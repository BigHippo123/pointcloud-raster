#!/usr/bin/env python3
"""
Minimal GPU benchmark test to isolate crash.
"""

import sys
sys.path.insert(0, '/workspace/python')

import os
import shutil
import pcr
from pcr.test_generators import generate_uniform_grid

def main():
    print("Minimal GPU Benchmark Test")
    print("="*80)

    # Create small grid (10M points)
    print("\nStep 1: Configure grid...")
    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 3162.0  # ~10M cells
    bbox.max_y = 3163.0

    grid_config = pcr.GridConfig()
    grid_config.bounds = bbox
    grid_config.crs = pcr.CRS.from_epsg(32610)
    grid_config.cell_size_x = 1.0
    grid_config.cell_size_y = -1.0
    grid_config.compute_dimensions()

    print(f"  Grid: {grid_config.width} × {grid_config.height} = {grid_config.width * grid_config.height:,} cells")
    print(f"  Tiles: {grid_config.tiles_x} × {grid_config.tiles_y}")

    # Setup pipeline
    print("\nStep 2: Create pipeline...")
    state_dir = "/tmp/test_gpu_simple"
    output_path = "/tmp/test_gpu_simple.tif"

    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.makedirs(state_dir, exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    reduction = pcr.ReductionSpec()
    reduction.value_channel = "value"
    reduction.type = pcr.ReductionType.Average
    reduction.output_band_name = "test"

    config = pcr.PipelineConfig()
    config.grid = grid_config
    config.reductions = [reduction]
    config.output_path = output_path
    config.state_dir = state_dir
    config.exec_mode = pcr.ExecutionMode.Auto  # Use Auto mode
    config.gpu_memory_budget = 0

    print("  Creating pipeline...")
    pipeline = pcr.Pipeline.create(config)
    print("  ✓ Pipeline created")

    # Generate points in Host memory
    print("\nStep 3: Generate points...")
    cloud, _ = generate_uniform_grid(bbox, 1.0, points_per_cell=1, value=50.0)
    print(f"  Generated: {cloud.count():,} points")
    print(f"  Location: {cloud.location()}")

    # Transfer to Device
    print("\nStep 4: Transfer to Device...")
    try:
        device_cloud = cloud.to_device()
        print(f"  ✓ Transfer successful: {device_cloud.count():,} points on Device")
    except Exception as e:
        print(f"  ✗ Transfer failed: {e}")
        return 1

    # Ingest
    print("\nStep 5: Ingest points...")
    try:
        pipeline.ingest(device_cloud)
        print(f"  ✓ Ingest successful")
    except Exception as e:
        print(f"  ✗ Ingest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Finalize
    print("\nStep 6: Finalize...")
    try:
        pipeline.finalize()
        print(f"  ✓ Finalize successful")
    except Exception as e:
        print(f"  ✗ Finalize failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n✓ Test complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
