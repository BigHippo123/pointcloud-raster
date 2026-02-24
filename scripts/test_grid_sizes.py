#!/usr/bin/env python3
"""
Test different grid sizes to find where 'cell index out of range' error occurs.
"""

import os
import sys
import shutil
sys.path.insert(0, '/workspace/python')
import pcr
from pcr.test_generators import generate_uniform_grid

def test_grid_size(size, points_per_cell=10):
    """Test a specific grid size."""
    print(f"\n{'='*70}")
    print(f"Testing Grid: {size} × {size}")
    print(f"{'='*70}")

    # Create bbox
    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = float(size)
    bbox.max_y = float(size)

    # Configure grid (exact match to working script)
    grid_config = pcr.GridConfig()
    grid_config.bounds = bbox
    grid_config.crs = pcr.CRS.from_epsg(32610)
    grid_config.cell_size_x = 1.0
    grid_config.cell_size_y = -1.0  # Negative!
    grid_config.compute_dimensions()

    print(f"  Grid dimensions: {grid_config.width} × {grid_config.height}")
    print(f"  Cell size: {grid_config.cell_size_x} × {grid_config.cell_size_y}")
    print(f"  Total cells: {grid_config.width * grid_config.height:,}")

    # Generate points using proven test generator
    print(f"  Generating {grid_config.width * grid_config.height * points_per_cell:,} points...")
    cloud, _ = generate_uniform_grid(
        bbox=bbox,
        cell_size=1.0,
        points_per_cell=points_per_cell,
        value=50.0,
        seed=42
    )

    print(f"  Generated: {cloud.count():,} points")

    # Configure pipeline
    state_dir = f"/tmp/pcr_test_grid_{size}"
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
    config.output_path = f"/tmp/test_grid_{size}.tif"
    config.state_dir = state_dir
    # CPU mode (no gpu_memory_budget set)

    print(f"  Creating pipeline...")
    pipeline = pcr.Pipeline.create(config)

    print(f"  Ingesting points...")
    try:
        pipeline.ingest(cloud)
        print(f"  ✓ Ingest successful!")

        print(f"  Finalizing...")
        pipeline.finalize()
        print(f"  ✓ Pipeline finalized!")

        stats = pipeline.stats()
        print(f"  ✓ Processed {stats.points_processed:,} points successfully")

        return True

    except RuntimeError as e:
        print(f"  ✗ ERROR: {e}")
        return False


def main():
    print("="*70)
    print("GRID SIZE INVESTIGATION")
    print("="*70)
    print("Testing incrementally larger grids to find breaking point...")

    # Test sizes (start with known working, then increase)
    test_sizes = [
        160,   # Known to work from generate_gpu_patterns.py
        200,   # Intermediate
        250,   # Intermediate
        300,   # Intermediate
        316,   # Currently failing
        400,   # Larger
        500,   # Even larger
    ]

    results = {}

    for size in test_sizes:
        success = test_grid_size(size, points_per_cell=10)
        results[size] = success

        if not success:
            print(f"\n{'='*70}")
            print(f"BREAKING POINT FOUND: Grid size {size} FAILS")
            print(f"{'='*70}")

            # Try to find exact breakpoint with binary search
            if size > 160:
                prev_size = [s for s in test_sizes if s < size][-1]
                print(f"\nBinary search between {prev_size} and {size}...")

                low = prev_size
                high = size

                while high - low > 1:
                    mid = (low + high) // 2
                    print(f"\nTesting size {mid}...")
                    if test_grid_size(mid, points_per_cell=10):
                        low = mid
                        print(f"  -> {mid} works, trying higher...")
                    else:
                        high = mid
                        print(f"  -> {mid} fails, trying lower...")

                print(f"\n{'='*70}")
                print(f"EXACT BREAKING POINT: Between {low} and {high}")
                print(f"  Last working size: {low} × {low}")
                print(f"  First failing size: {high} × {high}")
                print(f"{'='*70}")

            break

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for size, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {size:4d} × {size:4d}: {status}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
