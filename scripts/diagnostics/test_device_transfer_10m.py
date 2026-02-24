#!/usr/bin/env python3
"""
Test device memory transfer with 10M points.
"""

import sys
sys.path.insert(0, '/workspace/python')

import pcr
from pcr.test_generators import generate_uniform_grid

def main():
    print("Testing Device memory transfer with ~10M points...")

    # Create bbox for ~10M points
    # 10,000 × 10,000 = 100M cells, with 10% fill = 10M points
    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 10000.0  # 10km
    bbox.max_y = 10000.0  # 10km

    print(f"\nStep 1: Generate points in Host memory...")
    cloud, _ = generate_uniform_grid(bbox, 1.0, points_per_cell=1, value=50.0)
    print(f"  Generated: {cloud.count():,} points")
    print(f"  Location: {cloud.location()}")
    print(f"  Memory: ~{cloud.count() * 32 / 1024 / 1024:.1f} MB")

    print(f"\nStep 2: Transfer to Device memory...")
    try:
        device_cloud = cloud.to_device()
        if device_cloud:
            print(f"  ✓ Transfer successful!")
            print(f"  Points: {device_cloud.count():,}")
            print(f"  Location: {device_cloud.location()}")
        else:
            print(f"  ✗ Transfer failed: returned None")
            return 1
    except Exception as e:
        print(f"  ✗ Exception during transfer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\n✓ Test complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
