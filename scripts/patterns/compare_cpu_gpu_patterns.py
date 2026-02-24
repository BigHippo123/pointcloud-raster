#!/usr/bin/env python3
"""
Compare CPU vs GPU pattern outputs and report differences.
"""

import os
import sys
import argparse
import numpy as np

try:
    import rasterio
except ImportError:
    print("Error: rasterio not installed. Install with: pip install rasterio")
    sys.exit(1)


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


def compare_geotiff(cpu_path, gpu_path, tolerance=1e-4):
    """
    Compare two GeoTIFF files and return comparison metrics.

    Args:
        cpu_path: Path to CPU-generated GeoTIFF
        gpu_path: Path to GPU-generated GeoTIFF
        tolerance: Maximum acceptable difference

    Returns:
        dict: Comparison metrics (max_diff, mean_diff, num_different, etc.)
    """
    # Read both files
    with rasterio.open(cpu_path) as cpu_ds:
        cpu_data = cpu_ds.read(1)  # Read first band
        cpu_meta = cpu_ds.meta

    with rasterio.open(gpu_path) as gpu_ds:
        gpu_data = gpu_ds.read(1)  # Read first band
        gpu_meta = gpu_ds.meta

    # Check metadata consistency
    meta_match = (
        cpu_meta['width'] == gpu_meta['width'] and
        cpu_meta['height'] == gpu_meta['height'] and
        cpu_meta['crs'] == gpu_meta['crs']
    )

    # Handle NaN values (cells with no points)
    cpu_valid = ~np.isnan(cpu_data)
    gpu_valid = ~np.isnan(gpu_data)

    # Check if NaN patterns match
    nan_match = np.array_equal(cpu_valid, gpu_valid)

    # Calculate differences only where both have valid data
    valid_mask = cpu_valid & gpu_valid
    num_valid_cells = np.sum(valid_mask)

    if num_valid_cells == 0:
        # Both outputs are entirely NaN
        return {
            'max_diff': 0.0,
            'mean_diff': 0.0,
            'num_different': 0,
            'total_cells': cpu_data.size,
            'valid_cells': 0,
            'percent_different': 0.0,
            'meta_match': meta_match,
            'nan_match': nan_match,
            'status': 'PASS'
        }

    # Calculate absolute differences
    diff = np.abs(cpu_data[valid_mask] - gpu_data[valid_mask])

    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    # Count cells exceeding tolerance
    num_different = np.sum(diff > tolerance)
    percent_different = (num_different / num_valid_cells) * 100

    # Determine status
    status = 'PASS' if (max_diff <= tolerance and meta_match and nan_match) else 'FAIL'

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'num_different': num_different,
        'total_cells': cpu_data.size,
        'valid_cells': num_valid_cells,
        'percent_different': percent_different,
        'meta_match': meta_match,
        'nan_match': nan_match,
        'status': status
    }


def format_metric(value, precision=6):
    """Format a metric value with scientific notation if needed."""
    if value == 0:
        return "0.000000"
    elif value < 1e-3:
        return f"{value:.2e}"
    else:
        return f"{value:.{precision}f}"


def print_comparison_result(pattern_name, metrics):
    """Print formatted comparison result for a single pattern."""
    status = metrics['status']
    status_symbol = f"{Colors.GREEN}✓{Colors.ENDC}" if status == 'PASS' else f"{Colors.RED}✗{Colors.ENDC}"

    print(f"\n{pattern_name}:")
    print(f"  Max difference:     {format_metric(metrics['max_diff'])}")
    print(f"  Mean difference:    {format_metric(metrics['mean_diff'])}")
    print(f"  Different pixels:   {metrics['num_different']} ({metrics['percent_different']:.2f}%)")
    print(f"  Valid cells:        {metrics['valid_cells']:,} / {metrics['total_cells']:,}")
    print(f"  Metadata match:     {metrics['meta_match']}")
    print(f"  NaN pattern match:  {metrics['nan_match']}")
    print(f"  Status:             {status_symbol} {status}")


def main():
    parser = argparse.ArgumentParser(
        description='Compare CPU vs GPU pattern outputs'
    )
    parser.add_argument(
        '--tolerance',
        type=float,
        default=1e-4,
        help='Maximum acceptable difference (default: 1e-4)'
    )
    parser.add_argument(
        '--cpu-dir',
        type=str,
        default='pattern_outputs',
        help='Directory containing CPU-generated patterns (default: pattern_outputs)'
    )
    parser.add_argument(
        '--gpu-dir',
        type=str,
        default='gpu_pattern_outputs',
        help='Directory containing GPU-generated patterns (default: gpu_pattern_outputs)'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default=None,
        help='Compare only this pattern (e.g., "01_checkerboard")'
    )

    args = parser.parse_args()

    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Pattern Comparison: CPU vs GPU{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"Tolerance: {args.tolerance}")
    print(f"CPU directory: {args.cpu_dir}")
    print(f"GPU directory: {args.gpu_dir}")

    # Define all pattern names
    pattern_names = [
        "01_checkerboard",
        "02_stripes_horizontal",
        "03_stripes_vertical",
        "04_stripes_diagonal",
        "05_bullseye",
        "06_gradient_linear",
        "07_gradient_radial",
        "08_text_pcr",
        "09_shape_circle",
        "10_shape_square",
        "11_shape_triangle",
        "12_uniform_grid",
        "13_gaussian_clusters",
        "14_planar_surface",
    ]

    # Filter to single pattern if specified
    if args.pattern:
        if args.pattern in pattern_names:
            pattern_names = [args.pattern]
        else:
            print(f"{Colors.RED}Error: Pattern '{args.pattern}' not found{Colors.ENDC}")
            print(f"Available patterns: {', '.join(pattern_names)}")
            sys.exit(1)

    # Compare all patterns
    results = {}
    missing_files = []

    for pattern_name in pattern_names:
        cpu_file = os.path.join(args.cpu_dir, f"{pattern_name}.tif")
        gpu_file = os.path.join(args.gpu_dir, f"{pattern_name}.tif")

        # Check if files exist
        if not os.path.exists(cpu_file):
            missing_files.append(f"CPU: {cpu_file}")
            continue
        if not os.path.exists(gpu_file):
            missing_files.append(f"GPU: {gpu_file}")
            continue

        try:
            metrics = compare_geotiff(cpu_file, gpu_file, tolerance=args.tolerance)
            results[pattern_name] = metrics
            print_comparison_result(pattern_name, metrics)
        except Exception as e:
            print(f"\n{Colors.RED}Error comparing {pattern_name}: {e}{Colors.ENDC}")
            results[pattern_name] = {'status': 'ERROR'}

    # Print missing files
    if missing_files:
        print(f"\n{Colors.YELLOW}Missing files:{Colors.ENDC}")
        for missing in missing_files:
            print(f"  {missing}")

    # Print summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Summary{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")

    passed = sum(1 for m in results.values() if m['status'] == 'PASS')
    failed = sum(1 for m in results.values() if m['status'] == 'FAIL')
    errors = sum(1 for m in results.values() if m['status'] == 'ERROR')
    total = len(results)

    print(f"Total patterns: {total}")
    print(f"{Colors.GREEN}Passed: {passed}{Colors.ENDC}")
    if failed > 0:
        print(f"{Colors.RED}Failed: {failed}{Colors.ENDC}")
    if errors > 0:
        print(f"{Colors.YELLOW}Errors: {errors}{Colors.ENDC}")

    # Overall result
    if failed > 0 or errors > 0:
        print(f"\n{Colors.RED}{Colors.BOLD}OVERALL: FAIL{Colors.ENDC}")
        sys.exit(1)
    elif passed == total and total > 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}OVERALL: PASS ✓{Colors.ENDC}")
        sys.exit(0)
    else:
        print(f"\n{Colors.YELLOW}{Colors.BOLD}OVERALL: INCOMPLETE{Colors.ENDC}")
        sys.exit(1)


if __name__ == "__main__":
    main()
