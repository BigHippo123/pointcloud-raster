#!/usr/bin/env python3
"""
Diagnose why GPU is not providing speedup.
"""

import os
import sys
import shutil
import time
sys.path.insert(0, '/workspace/python')
import pcr
from pcr.test_generators import generate_uniform_grid

def test_execution_mode(exec_mode_name):
    """Test a specific execution mode configuration."""
    print(f"\n{'='*70}")
    print(f"Testing: {exec_mode_name}")
    print(f"{'='*70}")

    # Create small grid
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
    grid_config.compute_dimensions()

    print(f"Grid: {grid_config.width} × {grid_config.height}")

    # Generate points (Host memory)
    cloud, _ = generate_uniform_grid(bbox, 1.0, points_per_cell=10, value=50.0, seed=42)
    print(f"Points: {cloud.count():,} in {cloud.location()} memory")

    # Configure pipeline
    state_dir = f"/tmp/pcr_diag_{exec_mode_name.lower()}"
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
    config.output_path = f"/tmp/diag_{exec_mode_name.lower()}.tif"
    config.state_dir = state_dir

    # Try different configurations
    if exec_mode_name == "GPU_via_budget":
        config.gpu_memory_budget = 0
        print("Config: gpu_memory_budget = 0")
    elif exec_mode_name == "GPU_via_exec_mode":
        config.exec_mode = pcr.ExecutionMode.GPU
        config.gpu_memory_budget = 0
        print("Config: exec_mode = GPU, gpu_memory_budget = 0")
    elif exec_mode_name == "Auto_with_budget":
        config.exec_mode = pcr.ExecutionMode.Auto
        config.gpu_memory_budget = 0
        print("Config: exec_mode = Auto, gpu_memory_budget = 0")
    elif exec_mode_name == "CPU":
        config.exec_mode = pcr.ExecutionMode.CPU
        print("Config: exec_mode = CPU")
    else:
        print("Config: defaults")

    # Run
    start = time.time()
    try:
        pipeline = pcr.Pipeline.create(config)
        pipeline.ingest(cloud)
        pipeline.finalize()
        elapsed = time.time() - start

        stats = pipeline.stats()
        print(f"✓ Success: {stats.points_processed:,} points in {elapsed:.2f}s "
              f"({stats.points_processed/elapsed/1e6:.2f} Mpts/s)")

    except Exception as e:
        print(f"✗ ERROR: {e}")


def main():
    print("="*70)
    print("GPU EXECUTION MODE DIAGNOSTIC")
    print("="*70)

    # Test all configurations
    test_execution_mode("CPU")
    test_execution_mode("GPU_via_budget")
    test_execution_mode("GPU_via_exec_mode")
    test_execution_mode("Auto_with_budget")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print("If GPU_via_exec_mode crashes but GPU_via_budget works,")
    print("then ExecutionMode.GPU has a bug with Host memory points.")
    print("If GPU_via_budget has same performance as CPU,")
    print("then gpu_memory_budget alone doesn't enable GPU mode.")


if __name__ == "__main__":
    main()
