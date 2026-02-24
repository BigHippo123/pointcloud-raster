#!/usr/bin/env python3
"""Generate visualizations from benchmark GeoTIFF outputs."""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import pcr

def read_geotiff(path):
    """Read a GeoTIFF file and return the data array using PCR library."""
    try:
        # Use PCR's Grid to read the GeoTIFF
        grid = pcr.Grid.from_file(path)
        # Get the first band
        arr = grid.band_array(0).copy()
        return arr
    except Exception as e:
        print(f"Error reading {path}: {e}")
        # Try basic approach with rasterio if available
        try:
            import rasterio
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.float32)
                nodata = src.nodata
                if nodata is not None:
                    arr[arr == nodata] = np.nan
                return arr
        except ImportError:
            raise ValueError(f"Could not read {path} - install rasterio or ensure PCR Grid works")

def create_visualization(tif_path, png_path, title, cmap='terrain', max_px=2000):
    """Create a visualization from a GeoTIFF file."""
    print(f"Generating: {png_path}")

    # Read data
    arr = read_geotiff(tif_path)

    # Downsample if needed
    h, w = arr.shape
    factor = max(1, int(np.ceil(max(h, w) / max_px)))
    if factor > 1:
        arr = arr[::factor, ::factor]
        print(f"  Downsampled {w}×{h} → {arr.shape[1]}×{arr.shape[0]} (factor {factor}×)")

    # Calculate statistics
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        print(f"  Warning: No valid data in {tif_path}")
        return

    coverage = 100 * len(valid) / arr.size
    vmin = float(np.percentile(valid, 2))
    vmax = float(np.percentile(valid, 98))

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')

    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='equal')

    ax.set_title(f'{title}\n{coverage:.1f}% coverage | '
                f'Range: {valid.min():.1f}–{valid.max():.1f}m',
                fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('Elevation (m)', fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    # Grid info
    textstr = f'Grid: {arr.shape[1]}×{arr.shape[0]} (downsampled)' if factor > 1 else f'Grid: {w}×{h}'
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path}")

def create_comparison_grid(tif_paths, labels, png_path, cmap='terrain', max_px=1500):
    """Create a grid comparison of multiple GeoTIFFs."""
    print(f"Generating comparison: {png_path}")

    n = len(tif_paths)
    if n == 0:
        return

    # Determine grid layout
    if n <= 2:
        rows, cols = 1, n
    elif n <= 4:
        rows, cols = 2, 2
    elif n <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3

    # Read all arrays
    arrays = []
    for tif_path in tif_paths:
        if not os.path.exists(tif_path):
            print(f"  Warning: {tif_path} not found, skipping")
            continue
        arr = read_geotiff(tif_path)

        # Downsample
        h, w = arr.shape
        factor = max(1, int(np.ceil(max(h, w) / max_px)))
        if factor > 1:
            arr = arr[::factor, ::factor]

        arrays.append(arr)

    if len(arrays) == 0:
        print("  No valid arrays to plot")
        return

    # Get global color limits from first array (Point glyph)
    valid_ref = arrays[0][~np.isnan(arrays[0])]
    if len(valid_ref) == 0:
        print("  No valid data in reference array")
        return

    vmin = float(np.percentile(valid_ref, 2))
    vmax = float(np.percentile(valid_ref, 98))

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), facecolor='white')
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (arr, label) in enumerate(zip(arrays, labels[:len(arrays)])):
        ax = axes[idx]

        valid = arr[~np.isnan(arr)]
        coverage = 100 * len(valid) / arr.size if arr.size > 0 else 0

        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation='nearest', aspect='equal')

        ax.set_title(f'{label}\n{coverage:.1f}% coverage',
                    fontsize=11, fontweight='bold')
        ax.axis('off')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Elevation (m)', fontsize=9)
        cbar.ax.tick_params(labelsize=8)

    # Hide unused subplots
    for idx in range(len(arrays), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {png_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize benchmark GeoTIFF outputs')
    parser.add_argument('--results-dir', default='/workspace/benchmark_results')
    parser.add_argument('--output-dir', default='/workspace/benchmark_results/visualizations')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Define visualizations to create
    tiffs = {
        'cpu-1t': ('dc_lidar_cpu-1t_point.tif', 'CPU Single-Thread'),
        'cpu-mt': ('dc_lidar_cpu-mt_point.tif', 'CPU Multi-Thread'),
        'gpu': ('dc_lidar_gpu_point.tif', 'GPU'),
        'hybrid': ('dc_lidar_hybrid_point.tif', 'Hybrid'),
    }

    # Individual visualizations
    for key, (tif_name, title) in tiffs.items():
        tif_path = os.path.join(args.results_dir, tif_name)
        if os.path.exists(tif_path):
            png_path = os.path.join(args.output_dir, f'{key}_point.png')
            create_visualization(tif_path, png_path, title)

    # Mode comparison grid
    mode_tifs = []
    mode_labels = []
    for key, (tif_name, title) in tiffs.items():
        tif_path = os.path.join(args.results_dir, tif_name)
        if os.path.exists(tif_path):
            mode_tifs.append(tif_path)
            mode_labels.append(title)

    if mode_tifs:
        comparison_path = os.path.join(args.output_dir, 'mode_comparison.png')
        create_comparison_grid(mode_tifs, mode_labels, comparison_path)

    # Glyph comparison
    glyph_configs = [
        ('dc_lidar_cpu-mt_point.tif', 'Point Glyph'),
        ('dc_lidar_cpu-mt_line-2.tif', 'Line Glyph (2 cells)'),
        ('dc_lidar_cpu-mt_gaussian-1.tif', 'Gaussian σ=1'),
        ('dc_lidar_cpu-mt_gaussian-3.tif', 'Gaussian σ=3'),
        ('dc_lidar_cpu-mt_gaussian-5.tif', 'Gaussian σ=5'),
    ]

    glyph_tifs = []
    glyph_labels = []
    for tif_name, label in glyph_configs:
        tif_path = os.path.join(args.results_dir, tif_name)
        if os.path.exists(tif_path):
            glyph_tifs.append(tif_path)
            glyph_labels.append(label)

    if glyph_tifs:
        glyph_path = os.path.join(args.output_dir, 'glyph_comparison.png')
        create_comparison_grid(glyph_tifs, glyph_labels, glyph_path)

    print("\nVisualization complete!")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()
