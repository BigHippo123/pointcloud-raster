#!/usr/bin/env python3
"""Create side-by-side Point vs Gaussian comparison."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio

# Read both rasters
with rasterio.open('/workspace/benchmark_results/full_dataset/dc_lidar_gpu_point.tif') as src:
    point_arr = src.read(1).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        point_arr[point_arr == nodata] = np.nan

with rasterio.open('/workspace/benchmark_results/full_dataset/dc_lidar_gpu_gaussian-3.tif') as src:
    gaussian_arr = src.read(1).astype(np.float32)
    nodata = src.nodata
    if nodata is not None:
        gaussian_arr[gaussian_arr == nodata] = np.nan

# Downsample for visualization
factor = 3
point_d = point_arr[::factor, ::factor]
gaussian_d = gaussian_arr[::factor, ::factor]

# Calculate statistics
point_valid = point_d[~np.isnan(point_d)]
gaussian_valid = gaussian_d[~np.isnan(gaussian_d)]

point_coverage = 100 * len(point_valid) / point_d.size
gaussian_coverage = 100 * len(gaussian_valid) / gaussian_d.size

# Color limits from Point glyph
vmin = float(np.percentile(point_valid, 2))
vmax = float(np.percentile(point_valid, 98))

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

# Point glyph
im1 = ax1.imshow(point_d, cmap='terrain', vmin=vmin, vmax=vmax,
                interpolation='nearest', aspect='equal')
ax1.set_title(f'Point Glyph\n{point_coverage:.1f}% coverage | Sharp, sparse',
             fontsize=14, fontweight='bold', pad=15)
ax1.axis('off')
cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, shrink=0.8)
cbar1.set_label('Elevation (m)', fontsize=11)

# Gaussian glyph
im2 = ax2.imshow(gaussian_d, cmap='terrain', vmin=vmin, vmax=vmax,
                interpolation='nearest', aspect='equal')
ax2.set_title(f'Gaussian Glyph (σ=3 cells)\n{gaussian_coverage:.1f}% coverage | Smooth, gap-filled',
             fontsize=14, fontweight='bold', pad=15)
ax2.axis('off')
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, shrink=0.8)
cbar2.set_label('Elevation (m)', fontsize=11)

plt.suptitle('Point vs Gaussian Glyph Comparison — DC LiDAR (GPU Mode, 10 files, 30.3M points)',
            fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('/workspace/benchmark_results/full_dataset/visualizations/point_vs_gaussian_comparison.png',
           dpi=150, bbox_inches='tight', facecolor='white')
print("Saved: point_vs_gaussian_comparison.png")
print(f"Point coverage: {point_coverage:.1f}%")
print(f"Gaussian coverage: {gaussian_coverage:.1f}%")
print(f"Coverage increase: {gaussian_coverage - point_coverage:.1f}%")
