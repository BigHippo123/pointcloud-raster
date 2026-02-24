#!/usr/bin/env python3
"""
Debug GPU crash with detailed logging.
"""

import sys
print("DEBUG: Starting script...")
sys.path.insert(0, '/workspace/python')

print("DEBUG: Importing pcr...")
import pcr
print("DEBUG: pcr imported successfully")

print("DEBUG: Importing test_generators...")
from pcr.test_generators import generate_uniform_grid
print("DEBUG: test_generators imported successfully")

import os
import shutil

def main():
    print("DEBUG: Entering main()")

    # Test 1: Just create a simple grid config
    print("\nTest 1: Create GridConfig...")
    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 1000.0
    bbox.max_y = 1000.0

    grid_config = pcr.GridConfig()
    grid_config.bounds = bbox
    grid_config.crs = pcr.CRS.from_epsg(32610)
    grid_config.cell_size_x = 1.0
    grid_config.cell_size_y = -1.0
    print("DEBUG: Calling compute_dimensions()...")
    grid_config.compute_dimensions()
    print(f"✓ Grid config created: {grid_config.width} × {grid_config.height}")

    # Test 2: Create PipelineConfig
    print("\nTest 2: Create PipelineConfig...")
    config = pcr.PipelineConfig()
    config.grid = grid_config
    print("✓ PipelineConfig created")

    # Test 3: Add reduction
    print("\nTest 3: Add reduction...")
    reduction = pcr.ReductionSpec()
    reduction.value_channel = "value"
    reduction.type = pcr.ReductionType.Average
    reduction.output_band_name = "test"
    config.reductions = [reduction]
    print("✓ Reduction added")

    # Test 4: Set paths
    print("\nTest 4: Set output paths...")
    state_dir = "/tmp/test_crash_debug"
    output_path = "/tmp/test_crash_debug.tif"

    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.makedirs(state_dir, exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    config.output_path = output_path
    config.state_dir = state_dir
    print("✓ Paths set")

    # Test 5: Set execution mode
    print("\nTest 5: Set execution mode...")
    config.exec_mode = pcr.ExecutionMode.Auto
    config.gpu_memory_budget = 0
    print("✓ Execution mode set")

    # Test 6: Create pipeline
    print("\nTest 6: Create Pipeline...")
    print("DEBUG: About to call Pipeline.create()...")
    try:
        pipeline = pcr.Pipeline.create(config)
        print("✓ Pipeline created successfully!")
    except Exception as e:
        print(f"✗ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 7: Generate points
    print("\nTest 7: Generate points...")
    try:
        cloud, _ = generate_uniform_grid(bbox, 1.0, points_per_cell=1, value=50.0)
        print(f"✓ Generated: {cloud.count():,} points")
    except Exception as e:
        print(f"✗ Point generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 8: Transfer to Device
    print("\nTest 8: Transfer to Device...")
    try:
        device_cloud = cloud.to_device()
        print(f"✓ Transfer successful: {device_cloud.count():,} points")
    except Exception as e:
        print(f"✗ Device transfer failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 9: Ingest
    print("\nTest 9: Ingest points...")
    try:
        pipeline.ingest(device_cloud)
        print(f"✓ Ingest successful")
    except Exception as e:
        print(f"✗ Ingest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Test 10: Finalize
    print("\nTest 10: Finalize...")
    try:
        pipeline.finalize()
        print(f"✓ Finalize successful")
    except Exception as e:
        print(f"✗ Finalize failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n✓ All tests passed!")
    return 0

if __name__ == "__main__":
    print("DEBUG: __main__ block")
    sys.exit(main())
