#!/usr/bin/env python3
"""
Smiley Face Point Cloud Rasterization Demo

Generates a point cloud in the shape of a smiley face and rasterizes it
to show the PCR library in action!
"""

import sys
import os
import numpy as np

# Add python module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../python'))

import pcr


def generate_circle_points(cx, cy, radius, num_points, intensity):
    """Generate points in a circle."""
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    # Vary the radius slightly for a filled circle
    radii = np.random.uniform(0, radius, num_points)
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)
    intensities = np.full(num_points, intensity, dtype=np.float32)
    return x, y, intensities


def generate_arc_points(cx, cy, radius, start_angle, end_angle, num_points, intensity, thickness=2.0):
    """Generate points along an arc (for the smile)."""
    angles = np.linspace(start_angle, end_angle, num_points)

    # Create thickness by adding random offset perpendicular to arc
    x_list = []
    y_list = []
    for angle in angles:
        for _ in range(3):  # Multiple points per angle for thickness
            r_offset = np.random.uniform(-thickness, thickness)
            r = radius + r_offset
            x_list.append(cx + r * np.cos(angle))
            y_list.append(cy + r * np.sin(angle))

    x = np.array(x_list)
    y = np.array(y_list)
    intensities = np.full(len(x), intensity, dtype=np.float32)
    return x, y, intensities


def generate_smiley_face_cloud():
    """Generate a point cloud in the shape of a smiley face."""
    print("Generating smiley face point cloud...")

    # Face dimensions (in a 100x100 grid)
    face_center_x = 50.0
    face_center_y = 50.0

    # Left eye (circle)
    left_eye_x = 35.0
    left_eye_y = 60.0
    left_eye_radius = 5.0

    # Right eye (circle)
    right_eye_x = 65.0
    right_eye_y = 60.0
    right_eye_radius = 5.0

    # Smile (arc)
    smile_center_y = 45.0
    smile_radius = 20.0
    smile_start_angle = -2.8  # radians (slightly past bottom)
    smile_end_angle = -0.34   # radians (slightly past bottom)

    # Generate points for each feature
    all_x = []
    all_y = []
    all_intensities = []

    # Left eye (high intensity)
    x, y, i = generate_circle_points(left_eye_x, left_eye_y, left_eye_radius, 500, 200.0)
    all_x.append(x)
    all_y.append(y)
    all_intensities.append(i)

    # Right eye (high intensity)
    x, y, i = generate_circle_points(right_eye_x, right_eye_y, right_eye_radius, 500, 200.0)
    all_x.append(x)
    all_y.append(y)
    all_intensities.append(i)

    # Smile (medium-high intensity)
    x, y, i = generate_arc_points(face_center_x, smile_center_y, smile_radius,
                                   smile_start_angle, smile_end_angle, 100, 180.0, thickness=1.5)
    all_x.append(x)
    all_y.append(y)
    all_intensities.append(i)

    # Optional: Add some background noise (low intensity)
    background_points = 2000
    bg_x = np.random.uniform(10, 90, background_points)
    bg_y = np.random.uniform(10, 90, background_points)
    bg_intensity = np.random.uniform(5, 30, background_points).astype(np.float32)
    all_x.append(bg_x)
    all_y.append(bg_y)
    all_intensities.append(bg_intensity)

    # Combine all points
    x_all = np.concatenate(all_x)
    y_all = np.concatenate(all_y)
    intensity_all = np.concatenate(all_intensities)

    total_points = len(x_all)
    print(f"✓ Generated {total_points} points")
    print(f"  - Eyes: {1000} points @ intensity 200")
    print(f"  - Smile: ~{300} points @ intensity 180")
    print(f"  - Background: {background_points} points @ intensity 5-30")

    # Create point cloud
    cloud = pcr.PointCloud.create(total_points)
    cloud.add_channel("intensity", pcr.DataType.Float32)

    # Set data
    cloud.set_x_array(x_all)
    cloud.set_y_array(y_all)
    cloud.set_channel_array_f32("intensity", intensity_all)
    cloud.set_crs(pcr.CRS.from_epsg(32633))

    return cloud


def create_ascii_preview(grid, band_idx=0):
    """Create an ASCII art preview of the rasterized result."""
    arr = grid.band_array(band_idx)

    # Normalize to 0-9 scale (ignore NaN)
    valid_mask = ~np.isnan(arr)
    if not valid_mask.any():
        return "No valid data"

    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)

    # Normalize
    arr_norm = (arr - vmin) / (vmax - vmin) if vmax > vmin else arr
    arr_norm = np.nan_to_num(arr_norm, nan=0)

    # Convert to character intensity
    chars = " .:-=+*#%@"
    arr_char = (arr_norm * (len(chars) - 1)).astype(int)
    arr_char = np.clip(arr_char, 0, len(chars) - 1)

    # Build ASCII image (flip Y for proper display)
    lines = []
    for row in arr_char:
        line = ''.join(chars[val] for val in row)
        lines.append(line)

    # Reverse to show Y-up
    lines.reverse()

    return '\n'.join(lines)


def main():
    """Main entry point."""
    print("=" * 70)
    print("🙂 SMILEY FACE POINT CLOUD RASTERIZATION DEMO 🙂")
    print("=" * 70)
    print()

    # 1. Generate smiley face point cloud
    cloud = generate_smiley_face_cloud()
    print(f"\nPoint cloud created: {cloud}")

    # 2. Configure pipeline
    print("\nConfiguring rasterization pipeline...")
    config = pcr.PipelineConfig()

    # Grid: 100x100 cells, 1m resolution
    config.grid.bounds.min_x = 0.0
    config.grid.bounds.min_y = 0.0
    config.grid.bounds.max_x = 100.0
    config.grid.bounds.max_y = 100.0
    config.grid.cell_size_x = 1.0
    config.grid.cell_size_y = -1.0
    config.grid.tile_width = 100  # Single tile for this small grid
    config.grid.tile_height = 100
    config.grid.crs = pcr.CRS.from_epsg(32633)
    config.grid.compute_dimensions()

    print(f"✓ Grid: {config.grid.width}x{config.grid.height} cells")

    # Reduction: Average intensity
    reduction = pcr.ReductionSpec()
    reduction.value_channel = "intensity"
    reduction.type = pcr.ReductionType.Average
    reduction.output_band_name = "smiley_intensity"

    # Set reductions (need to assign the list, not append)
    config.reductions = [reduction]

    # Output
    output_path = "/tmp/smiley_face.tif"
    config.output_path = output_path
    config.state_dir = "/tmp/smiley_tiles"
    config.write_cog = True

    print(f"✓ Reduction: Average intensity")
    print(f"✓ Output: {output_path}")

    # 3. Run pipeline
    print("\nRasterizing smiley face...")
    pipeline = pcr.Pipeline.create(config)
    pipeline.validate()

    # Progress callback
    def progress(info):
        if info.points_processed > 0:
            print(f"  → Processed {info.points_processed} points in {info.elapsed_seconds:.2f}s")
        return True

    pipeline.set_progress_callback(progress)

    # Process
    pipeline.ingest(cloud)
    pipeline.finalize()

    stats = pipeline.stats()
    print(f"✓ Rasterization complete!")
    print(f"  Total points: {stats.points_processed}")
    print(f"  Time: {stats.elapsed_seconds:.3f}s")

    # 4. Get and display result
    result = pipeline.result()
    print(f"\n✓ Result grid: {result}")

    # Statistics
    arr = result.band_array(0)
    valid_mask = ~np.isnan(arr)
    num_valid = np.sum(valid_mask)

    print(f"\nIntensity Statistics:")
    print(f"  Valid cells: {num_valid}/{arr.size} ({100*num_valid/arr.size:.1f}%)")
    if num_valid > 0:
        print(f"  Min: {np.nanmin(arr):.2f}")
        print(f"  Max: {np.nanmax(arr):.2f}")
        print(f"  Mean: {np.nanmean(arr):.2f}")

    # 5. ASCII art preview
    print("\n" + "=" * 70)
    print("ASCII ART PREVIEW (rotated 90° - view from the side!):")
    print("=" * 70)
    print()

    ascii_art = create_ascii_preview(result, 0)
    print(ascii_art)

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print(f"\nOutput saved to: {output_path}")
    print("\nView the result with:")
    print(f"  gdalinfo {output_path}")
    print(f"  gdal_translate -of PNG -scale {output_path} /tmp/smiley.png")
    print(f"  display /tmp/smiley.png  # or: xdg-open /tmp/smiley.png")
    print()
    print("🙂 Have a nice day! 🙂")
    print()


if __name__ == "__main__":
    main()
