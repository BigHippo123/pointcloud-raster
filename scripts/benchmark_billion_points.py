#!/usr/bin/env python3
"""
1 Billion Point CPU vs GPU Performance Benchmark

Processes up to 1 billion points through PCR pipeline to measure:
- CPU performance (baseline)
- GPU performance (accelerated)
- Speedup factor
- Memory usage
- Throughput (points/sec)

Usage:
    # Quick test with 10M points
    python benchmark_billion_points.py --points 10000000

    # Full 1B point benchmark
    python benchmark_billion_points.py --points 1000000000
"""

import os
import sys
import time
import shutil
import subprocess
import argparse
import numpy as np

# Add PCR Python bindings to path
sys.path.insert(0, '/workspace/python')
import pcr
from pcr.test_generators import generate_uniform_grid

# Optional psutil for memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_gpu_info():
    """Get GPU information via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def get_gpu_memory_used():
    """Query current GPU memory usage via nvidia-smi (in MB)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except:
        pass
    return None


def create_grid_config(total_points, points_per_cell=10, use_multitile=True):
    """
    Create grid configuration sized for target number of points.

    Strategy: For GPU performance, create multi-tile grids.
    Tile size is 4096x4096. GPU performs best with multiple tiles for parallelism.

    Args:
        total_points: Target total number of points
        points_per_cell: Average points per cell
        use_multitile: If True, create 2x2+ tile grid for GPU; if False, optimize for size

    Returns:
        (GridConfig, BBox, num_cells)
    """
    num_cells = total_points // points_per_cell

    if use_multitile and total_points >= 10_000_000:
        # GPU Mode: Create multi-tile grid for parallelism
        # Target: At least 2x2 tiles (4 tiles minimum)
        TILE_SIZE = 4096
        cells_per_tile = TILE_SIZE * TILE_SIZE  # 16,777,216 cells per tile

        # Calculate how many tiles we need
        tiles_needed = max(4, int(np.ceil(num_cells / cells_per_tile)))

        # Make it roughly square in tiles
        tiles_per_side = int(np.ceil(np.sqrt(tiles_needed)))

        # Grid dimensions in cells
        width = tiles_per_side * TILE_SIZE
        height = tiles_per_side * TILE_SIZE

        actual_cells = width * height

        print(f"Grid Configuration (Multi-Tile for GPU):")
        print(f"  Target cells: {num_cells:,}")
        print(f"  Actual cells: {actual_cells:,} ({width:,} × {height:,})")
        print(f"  Tile layout: {tiles_per_side} × {tiles_per_side} = {tiles_needed} tiles")
        print(f"  Cells per tile: {TILE_SIZE} × {TILE_SIZE} = {cells_per_tile:,}")
    else:
        # CPU Mode or Small Dataset: Optimize for exact size
        side = int(np.sqrt(num_cells))
        width = side
        height = side

        # Adjust to match target cells
        while width * height < num_cells:
            height += 1

        actual_cells = width * height

        print(f"Grid Configuration (Single-Tile):")
        print(f"  Target cells: {num_cells:,}")
        print(f"  Actual cells: {actual_cells:,} ({width:,} × {height:,})")

    print(f"Grid Configuration:")
    print(f"  Target cells: {num_cells:,}")
    print(f"  Actual cells: {actual_cells:,} ({width:,} × {height:,})")
    print(f"  Points per cell: {points_per_cell}")
    print(f"  Expected points: {actual_cells * points_per_cell:,}")

    # Create bounding box (1m cells)
    bbox = pcr.BBox()
    bbox.min_x = 0.0
    bbox.min_y = 0.0
    bbox.max_x = float(width)
    bbox.max_y = float(height)

    # Configure grid (match working script exactly)
    grid_config = pcr.GridConfig()
    grid_config.bounds = bbox
    grid_config.crs = pcr.CRS.from_epsg(32610)  # UTM Zone 10N
    grid_config.cell_size_x = 1.0
    grid_config.cell_size_y = -1.0  # CRITICAL: Must be negative!
    # Note: NOT setting tile_width/tile_height - use defaults
    grid_config.compute_dimensions()  # CRITICAL: Must call!

    # Verify configuration
    print(f"  Grid dimensions: {grid_config.width} × {grid_config.height}")
    print(f"  Cell size: {grid_config.cell_size_x} × {grid_config.cell_size_y}")
    print(f"  Bounds: ({bbox.min_x}, {bbox.min_y}) to ({bbox.max_x}, {bbox.max_y})")
    print(f"  Tile size: {grid_config.tile_width} × {grid_config.tile_height}")
    print(f"  Total tiles: {grid_config.tiles_x} × {grid_config.tiles_y} = "
          f"{grid_config.tiles_x * grid_config.tiles_y}")

    # Verify that grid dimensions match
    expected_width = int((bbox.max_x - bbox.min_x) / grid_config.cell_size_x)
    expected_height = int((bbox.max_y - bbox.min_y) / abs(grid_config.cell_size_y))
    print(f"  Expected dimensions: {expected_width} × {expected_height}")

    if grid_config.width != expected_width or grid_config.height != expected_height:
        print(f"  WARNING: Grid dimensions don't match expected!")

    return grid_config, bbox, actual_cells


def generate_point_chunk(bbox, points_per_chunk, chunk_id, seed=42, use_device_memory=False):
    """
    Generate a chunk of uniformly distributed points.

    Uses the proven generate_uniform_grid function from test_generators.

    Args:
        bbox: Bounding box for point generation
        points_per_chunk: Number of points to generate
        chunk_id: Chunk identifier for seeding
        seed: Base random seed
        use_device_memory: If True, transfer points to GPU Device memory

    Returns:
        PointCloud with generated points (in Host or Device memory)
    """
    # Calculate grid dimensions and points per cell
    width = bbox.max_x - bbox.min_x
    height = bbox.max_y - bbox.min_y
    total_cells = int(width * height)

    # Calculate how many points per cell we need
    points_per_cell = max(1, int(points_per_chunk / total_cells))

    # Use the proven test generator (creates points in Host memory)
    cloud, _ = generate_uniform_grid(
        bbox=bbox,
        cell_size=1.0,  # Match grid config
        points_per_cell=points_per_cell,
        value=50.0,
        seed=seed + chunk_id
    )

    # Transfer to Device memory if requested (for GPU mode)
    if use_device_memory:
        print(f"  Attempting to transfer {cloud.count():,} points to Device memory...")
        try:
            import traceback
            device_cloud = cloud.to_device()
            if device_cloud and device_cloud.location() == pcr.MemoryLocation.Device:
                print(f"  ✓ Transfer successful: {device_cloud.count():,} points on Device")
                return device_cloud
            else:
                print(f"  ✗ Transfer failed: returned {device_cloud}")
                return cloud
        except Exception as e:
            print(f"  ✗ Failed to transfer to Device memory: {e}")
            traceback.print_exc()
            print(f"  Falling back to Host memory")
            return cloud

    return cloud


def run_benchmark(grid_config, bbox, exec_mode, total_points,
                  chunk_size=100_000_000, seed=42):
    """
    Run pipeline benchmark for specified execution mode.

    Args:
        grid_config: Grid configuration
        bbox: Bounding box for point generation
        exec_mode: pcr.ExecutionMode.CPU or pcr.ExecutionMode.GPU
        total_points: Total number of points to process
        chunk_size: Points per chunk (to manage memory)
        seed: Random seed for reproducibility

    Returns:
        dict: Performance metrics
    """
    mode_name = "GPU" if exec_mode == pcr.ExecutionMode.GPU else "CPU"
    num_chunks = max(1, (total_points + chunk_size - 1) // chunk_size)

    print(f"\n{'='*80}")
    print(f"Running {mode_name} Benchmark")
    print(f"{'='*80}")
    print(f"Total points: {total_points:,}")
    print(f"Processing: {num_chunks} chunk(s) of ~{chunk_size:,} points each")

    # Setup pipeline configuration
    state_dir = f"/tmp/pcr_benchmark_{mode_name.lower()}"
    output_path = f"/tmp/benchmark_{mode_name.lower()}.tif"

    # Clean previous state
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.makedirs(state_dir, exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    # Configure reduction (simple average for performance testing)
    reduction = pcr.ReductionSpec()
    reduction.value_channel = "value"
    reduction.type = pcr.ReductionType.Average
    reduction.output_band_name = f"benchmark_{mode_name}"

    # Configure pipeline
    config = pcr.PipelineConfig()
    config.grid = grid_config
    config.reductions = [reduction]
    config.output_path = output_path
    config.state_dir = state_dir

    # Configure execution mode
    # For GPU: Use Auto mode with Device memory points (avoids crash bug)
    # For CPU: Use explicit CPU mode
    if exec_mode == pcr.ExecutionMode.GPU:
        config.exec_mode = pcr.ExecutionMode.Auto  # Auto will detect Device memory
        config.gpu_memory_budget = 0  # Auto-detect GPU memory
        print(f"  Mode: Auto (with Device memory points for GPU)")
        print(f"  GPU memory budget: auto-detect")
    else:
        config.exec_mode = pcr.ExecutionMode.CPU
        print(f"  Mode: CPU")

    print(f"  Execution mode: {mode_name}")
    print(f"  Output: {output_path}")

    # Create pipeline
    pipeline = pcr.Pipeline.create(config)

    # Metrics collection
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        start_ram = process.memory_info().rss / 1024 / 1024  # MB
    else:
        process = None
        start_ram = 0.0

    start_gpu_mem = get_gpu_memory_used()

    chunk_times = []
    total_points_processed = 0

    # Start timer
    start_time = time.time()

    # Process chunks
    for chunk_id in range(num_chunks):
        chunk_start = time.time()

        # Calculate points for this chunk
        remaining = total_points - total_points_processed
        points_this_chunk = min(chunk_size, remaining)

        print(f"\n[{mode_name}] Chunk {chunk_id + 1}/{num_chunks}: "
              f"Generating {points_this_chunk:,} points...")

        # Generate point chunk
        # For GPU mode, use Device memory to enable Auto mode GPU execution
        use_device = (exec_mode == pcr.ExecutionMode.GPU)
        cloud = generate_point_chunk(bbox, points_this_chunk, chunk_id, seed, use_device_memory=use_device)
        actual_points = cloud.count()
        total_points_processed += actual_points

        if chunk_id == 0:  # Log memory location for first chunk
            print(f"  Point cloud memory: {cloud.location()}")

        # Ingest chunk
        gen_time = time.time() - chunk_start
        ingest_start = time.time()
        pipeline.ingest(cloud)
        ingest_time = time.time() - ingest_start

        chunk_elapsed = time.time() - chunk_start
        chunk_times.append(ingest_time)  # Only count ingest time

        throughput = actual_points / ingest_time if ingest_time > 0 else 0
        print(f"  Generated: {actual_points:,} points in {gen_time:.2f}s")
        print(f"  Ingested: {actual_points:,} points in {ingest_time:.2f}s "
              f"({throughput/1e6:.2f} Mpts/s)")

    # Finalize pipeline
    print(f"\n[{mode_name}] Finalizing...")
    finalize_start = time.time()
    pipeline.finalize()
    finalize_time = time.time() - finalize_start
    print(f"  Finalized in {finalize_time:.2f}s")

    # Stop timer
    total_time = time.time() - start_time
    total_ingest_time = sum(chunk_times)

    # Collect final metrics
    if HAS_PSUTIL and process:
        current_ram = process.memory_info().rss / 1024 / 1024  # MB
        peak_ram = process.memory_info().rss / 1024 / 1024  # MB (simplified)
    else:
        current_ram = 0.0
        peak_ram = 0.0

    end_gpu_mem = get_gpu_memory_used()

    stats = pipeline.stats()

    # Calculate metrics
    throughput = stats.points_processed / total_ingest_time if total_ingest_time > 0 else 0

    results = {
        'mode': mode_name,
        'total_points': stats.points_processed,
        'num_chunks': num_chunks,
        'total_time_sec': total_time,
        'ingest_time_sec': total_ingest_time,
        'finalize_time_sec': finalize_time,
        'throughput_mpts': throughput / 1e6,
        'avg_chunk_time_sec': np.mean(chunk_times) if chunk_times else 0,
        'start_ram_mb': start_ram,
        'end_ram_mb': current_ram,
        'peak_ram_mb': peak_ram,
        'ram_increase_mb': current_ram - start_ram,
        'gpu_mem_start_mb': start_gpu_mem,
        'gpu_mem_end_mb': end_gpu_mem,
        'gpu_mem_used_mb': (end_gpu_mem - start_gpu_mem) if start_gpu_mem and end_gpu_mem else None
    }

    print(f"\n[{mode_name}] Summary:")
    print(f"  Points processed: {stats.points_processed:,}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Ingest time: {total_ingest_time:.2f}s")
    print(f"  Throughput: {throughput/1e6:.2f} Mpts/s")
    if HAS_PSUTIL:
        print(f"  RAM usage: {start_ram:.0f} MB → {current_ram:.0f} MB "
              f"(+{current_ram - start_ram:.0f} MB)")

    return results


def print_comparison(cpu_results, gpu_results):
    """Print formatted comparison results."""
    print(f"\n\n{'='*80}")
    print("PERFORMANCE COMPARISON RESULTS")
    print(f"{'='*80}\n")

    # Summary table
    print(f"{'Metric':<35} {'CPU':<20} {'GPU':<20} {'Speedup':<15}")
    print("-" * 95)

    # Total points
    cpu_pts = cpu_results['total_points']
    gpu_pts = gpu_results['total_points']
    print(f"{'Points Processed':<35} {cpu_pts:>19,} {gpu_pts:>19,}")

    # Ingest time (excludes point generation)
    cpu_time = cpu_results['ingest_time_sec']
    gpu_time = gpu_results['ingest_time_sec']
    speedup_ingest = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"{'Ingest Time (sec)':<35} {cpu_time:>19.2f} {gpu_time:>19.2f} "
          f"{speedup_ingest:>14.2f}x")

    # Total time (includes generation + finalization)
    cpu_total = cpu_results['total_time_sec']
    gpu_total = gpu_results['total_time_sec']
    speedup_total = cpu_total / gpu_total if gpu_total > 0 else 0
    print(f"{'Total Time (sec)':<35} {cpu_total:>19.2f} {gpu_total:>19.2f} "
          f"{speedup_total:>14.2f}x")

    # Throughput
    cpu_tput = cpu_results['throughput_mpts']
    gpu_tput = gpu_results['throughput_mpts']
    tput_ratio = gpu_tput / cpu_tput if cpu_tput > 0 else 0
    print(f"{'Throughput (Mpts/s)':<35} {cpu_tput:>19.2f} {gpu_tput:>19.2f} "
          f"{tput_ratio:>14.2f}x")

    # Memory
    print(f"{'Peak RAM (MB)':<35} {cpu_results['peak_ram_mb']:>19.0f} "
          f"{gpu_results['peak_ram_mb']:>19.0f}")

    if gpu_results['gpu_mem_used_mb'] is not None:
        print(f"{'GPU Memory Used (MB)':<35} {'N/A':>19} "
              f"{gpu_results['gpu_mem_used_mb']:>19.0f}")

    # Speedup summary
    print(f"\n{'='*80}")
    print(f"GPU SPEEDUP: {speedup_ingest:.2f}x faster than CPU (ingest)")
    print(f"GPU THROUGHPUT: {gpu_tput:.2f} million points/second")
    print(f"{'='*80}\n")


def save_results_csv(cpu_results, gpu_results, filename):
    """Save results to CSV file."""
    with open(filename, 'w') as f:
        f.write("Metric,CPU,GPU,Speedup\n")
        f.write(f"Points,{cpu_results['total_points']},{gpu_results['total_points']},1.0\n")

        cpu_time = cpu_results['ingest_time_sec']
        gpu_time = gpu_results['ingest_time_sec']
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        f.write(f"Ingest_Time_sec,{cpu_time:.3f},{gpu_time:.3f},{speedup:.2f}\n")

        cpu_total = cpu_results['total_time_sec']
        gpu_total = gpu_results['total_time_sec']
        speedup_total = cpu_total / gpu_total if gpu_total > 0 else 0
        f.write(f"Total_Time_sec,{cpu_total:.3f},{gpu_total:.3f},{speedup_total:.2f}\n")

        cpu_tput = cpu_results['throughput_mpts']
        gpu_tput = gpu_results['throughput_mpts']
        tput_ratio = gpu_tput / cpu_tput if cpu_tput > 0 else 0
        f.write(f"Throughput_Mpts,{cpu_tput:.2f},{gpu_tput:.2f},{tput_ratio:.2f}\n")
        f.write(f"Peak_RAM_MB,{cpu_results['peak_ram_mb']:.0f},{gpu_results['peak_ram_mb']:.0f},1.0\n")

        if gpu_results['gpu_mem_used_mb'] is not None:
            f.write(f"GPU_Memory_MB,0,{gpu_results['gpu_mem_used_mb']:.0f},N/A\n")

    print(f"✓ Results saved to: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark PCR CPU vs GPU performance on large point clouds'
    )
    parser.add_argument(
        '--points', type=int, default=10_000_000,
        help='Total number of points to process (default: 10M, use 1B for full benchmark)'
    )
    parser.add_argument(
        '--chunk-size', type=int, default=100_000_000,
        help='Points per chunk (default: 100M)'
    )
    parser.add_argument(
        '--output', type=str, default='/workspace/scripts/benchmark_results.csv',
        help='Output CSV file path'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--multitile', action='store_true', default=True,
        help='Use multi-tile grid strategy for GPU (default: True)'
    )
    parser.add_argument(
        '--single-tile', dest='multitile', action='store_false',
        help='Use single-tile grid (for comparison)'
    )

    args = parser.parse_args()

    print("="*80)
    print("PCR: CPU vs GPU Performance Benchmark")
    print("="*80)
    print(f"Target points: {args.points:,}")
    print(f"Chunk size: {args.chunk_size:,}")

    # Check GPU availability
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"\n✓ GPU Available: {gpu_info}")
    else:
        print("\n✗ GPU not detected - cannot run GPU benchmark")
        return 1

    # Create grid configuration
    print(f"\n{'='*80}")
    print("Grid Setup")
    print(f"{'='*80}")
    grid_config, bbox, num_cells = create_grid_config(
        args.points,
        points_per_cell=10,
        use_multitile=args.multitile
    )

    # Run CPU benchmark
    cpu_results = run_benchmark(
        grid_config, bbox,
        pcr.ExecutionMode.CPU,
        args.points,
        chunk_size=args.chunk_size,
        seed=args.seed
    )

    # Run GPU benchmark
    gpu_results = run_benchmark(
        grid_config, bbox,
        pcr.ExecutionMode.GPU,
        args.points,
        chunk_size=args.chunk_size,
        seed=args.seed
    )

    # Print comparison
    print_comparison(cpu_results, gpu_results)

    # Save to CSV
    save_results_csv(cpu_results, gpu_results, args.output)

    print(f"\n✓ Benchmark complete!")
    print(f"\nOutputs:")
    print(f"  CPU raster: /tmp/benchmark_cpu.tif")
    print(f"  GPU raster: /tmp/benchmark_gpu.tif")
    print(f"  Results CSV: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
