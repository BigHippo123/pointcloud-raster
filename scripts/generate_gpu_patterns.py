#!/usr/bin/env python3
"""
Generate all test patterns using GPU pipeline execution.
"""

import os
import pcr
from pcr.test_generators import (
    generate_checkerboard,
    generate_stripes,
    generate_bullseye,
    generate_gradient,
    generate_text,
    generate_shapes,
    generate_uniform_grid,
    generate_gaussian_clusters,
    generate_planar_surface,
)
from line_profiler import profile

# Create output directory
os.makedirs("gpu_pattern_outputs", exist_ok=True)

@profile
def process_pattern(name, cloud, metadata, cell_size=1.0):
    """Process a pattern through the GPU pipeline and save output."""
    print(f"\n{'='*70}")
    print(f"Processing: {name} (GPU)")
    print(f"{'='*70}")

    # Clean tile state to prevent contamination between patterns
    import shutil
    state_dir = "/tmp/pcr_tiles_gpu"
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.makedirs(state_dir, exist_ok=True)

    # Get bbox from metadata or create one
    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 160.0
    bbox.max_y = 160.0

    # Configure grid
    grid_config = pcr.GridConfig()
    grid_config.bounds = bbox
    grid_config.crs = pcr.CRS.from_epsg(32610)
    grid_config.cell_size_x = cell_size
    grid_config.cell_size_y = -cell_size
    grid_config.compute_dimensions()

    # Configure reduction
    reduction = pcr.ReductionSpec()
    reduction.value_channel = "value"
    reduction.type = pcr.ReductionType.Average
    reduction.output_band_name = name

    # Configure pipeline for GPU execution
    config = pcr.PipelineConfig()
    config.grid = grid_config
    config.reductions = [reduction]
    config.output_path = f"gpu_pattern_outputs/{name}.tif"
    config.exec_mode = pcr.ExecutionMode.GPU  # explicitly request GPU
    config.state_dir = state_dir

    # Transfer cloud to GPU before ingestion to avoid H2D overhead inside the pipeline
    device_cloud = cloud.to_device()

    print(f"  Point cloud: {device_cloud.count():,} points ({device_cloud.location()})")
    print(f"  Grid: {grid_config.width}x{grid_config.height} cells")
    print(f"  Output: {config.output_path}")

    # Run pipeline
    pipeline = pcr.Pipeline.create(config)
    pipeline.ingest(device_cloud)
    pipeline.finalize()

    stats = pipeline.stats()
    print(f"  ✓ Processed {stats.points_processed:,} points")
    print(f"  ✓ Output written to: {config.output_path}")

    return config.output_path


def main():
    print("="*70)
    print("GENERATING ALL TEST PATTERNS (GPU)")
    print("="*70)

    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = 160.0
    bbox.max_y = 160.0
    cell_size = 1.0

    patterns = []

    # -------------------------------------------------------------------------
    # Visual Patterns
    # -------------------------------------------------------------------------

    # 1. Checkerboard
    print("\n[1/14] Checkerboard pattern...")
    cloud, meta = generate_checkerboard(bbox, cell_size, points_per_cell=1_000, square_size=8)
    patterns.append(("01_checkerboard", cloud, meta))

    # 2. Horizontal stripes
    print("[2/14] Horizontal stripes...")
    cloud, meta = generate_stripes(bbox, cell_size, points_per_cell=20,
                                     stripe_width=5, orientation="horizontal")
    patterns.append(("02_stripes_horizontal", cloud, meta))

    # 3. Vertical stripes
    print("[3/14] Vertical stripes...")
    cloud, meta = generate_stripes(bbox, cell_size, points_per_cell=20,
                                     stripe_width=5, orientation="vertical")
    patterns.append(("03_stripes_vertical", cloud, meta))

    # 4. Diagonal stripes
    print("[4/14] Diagonal stripes...")
    cloud, meta = generate_stripes(bbox, cell_size, points_per_cell=20,
                                     stripe_width=5, orientation="diagonal")
    patterns.append(("04_stripes_diagonal", cloud, meta))

    # 5. Bullseye
    print("[5/14] Bullseye (concentric circles)...")
    cloud, meta = generate_bullseye(bbox, cell_size, points_per_cell=20, num_rings=8)
    patterns.append(("05_bullseye", cloud, meta))

    # 6. Linear gradient
    print("[6/14] Linear gradient...")
    cloud, meta = generate_gradient(bbox, cell_size, points_per_cell=20,
                                      gradient_type="linear", angle=0.0)
    patterns.append(("06_gradient_linear", cloud, meta))

    # 7. Radial gradient
    print("[7/14] Radial gradient...")
    cloud, meta = generate_gradient(bbox, cell_size, points_per_cell=20,
                                      gradient_type="radial")
    patterns.append(("07_gradient_radial", cloud, meta))

    # 8. Text pattern
    print("[8/14] Text pattern (PCR)...")
    cloud, meta = generate_text(bbox, cell_size, text="PCR", points_per_cell=20)
    patterns.append(("08_text_pcr", cloud, meta))

    # 9. Circle shape
    print("[9/14] Circle shape...")
    cloud, meta = generate_shapes(bbox, cell_size, shape="circle", points_per_cell=20)
    patterns.append(("09_shape_circle", cloud, meta))

    # 10. Square shape
    print("[10/14] Square shape...")
    cloud, meta = generate_shapes(bbox, cell_size, shape="square", points_per_cell=20)
    patterns.append(("10_shape_square", cloud, meta))

    # 11. Triangle shape
    print("[11/14] Triangle shape...")
    cloud, meta = generate_shapes(bbox, cell_size, shape="triangle", points_per_cell=20)
    patterns.append(("11_shape_triangle", cloud, meta))

    # -------------------------------------------------------------------------
    # Technical Patterns
    # -------------------------------------------------------------------------

    # 12. Uniform grid
    print("[12/14] Uniform grid (constant value)...")
    cloud, meta = generate_uniform_grid(bbox, cell_size, points_per_cell=10, value=75.0)
    patterns.append(("12_uniform_grid", cloud, meta))

    # 13. Gaussian clusters
    print("[13/14] Gaussian clusters...")
    cloud, meta = generate_gaussian_clusters(bbox, cell_size, num_clusters=8,
                                               points_per_cluster=2000, cluster_std=8.0)
    patterns.append(("13_gaussian_clusters", cloud, meta))

    # 14. Planar surface
    print("[14/14] Planar surface with noise...")
    cloud, meta = generate_planar_surface(bbox, cell_size, points_per_cell=10,
                                            slope_x=0.5, slope_y=0.3, noise_std=5.0)
    patterns.append(("14_planar_surface", cloud, meta))

    # -------------------------------------------------------------------------
    # Process all patterns through GPU pipeline
    # -------------------------------------------------------------------------

    print("\n" + "="*70)
    print("PROCESSING PATTERNS THROUGH GPU PIPELINE")
    print("="*70)

    outputs = []
    for name, cloud, meta in patterns:
        output_path = process_pattern(name, cloud, meta, cell_size)
        outputs.append((name, output_path))

    # -------------------------------------------------------------------------
    # Convert all to PNG
    # -------------------------------------------------------------------------

    print("\n" + "="*70)
    print("CONVERTING TO PNG FOR VIEWING")
    print("="*70)

    import subprocess

    for name, tif_path in outputs:
        png_path = tif_path.replace('.tif', '.png')

        # Use gdal_translate to convert to PNG
        cmd = [
            'gdal_translate',
            '-of', 'PNG',
            '-scale', '0', '100', '0', '255',
            tif_path,
            png_path
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  ✓ {name}.png")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to convert {name}: {e}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\n" + "="*70)
    print("✓ ALL GPU PATTERNS GENERATED!")
    print("="*70)
    print(f"\nOutput directory: gpu_pattern_outputs/")
    print(f"Generated {len(outputs)} patterns:")
    for name, _ in outputs:
        print(f"  - {name}.tif / {name}.png")

    print("\nView the images:")
    print("  ls -lh gpu_pattern_outputs/*.png")
    print("  eog gpu_pattern_outputs/*.png  # or your image viewer")


if __name__ == "__main__":
    main()
