#!/usr/bin/env python3
"""
Visualize CPU vs GPU performance comparison results.

Usage:
    python visualize_performance.py performance_results.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import numpy as np
from pathlib import Path

def load_results(csv_file):
    """Load performance results from CSV."""
    df = pd.read_csv(csv_file)
    return df

def plot_wall_time_comparison(df, output_dir):
    """Compare wall time between CPU and GPU."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, test in enumerate(df['test_name'].unique()):
        test_df = df[df['test_name'] == test]

        cpu_df = test_df[test_df['variant'] == 'CPU']
        gpu_df = test_df[test_df['variant'] == 'GPU']

        if len(cpu_df) == 0 or len(gpu_df) == 0:
            continue

        ax = axes[idx] if idx < 3 else axes[-1]

        x = range(len(cpu_df))
        width = 0.35

        ax.bar([i - width/2 for i in x], cpu_df['wall_time_ms'], width, label='CPU', alpha=0.8)
        ax.bar([i + width/2 for i in x], gpu_df['wall_time_ms'], width, label='GPU', alpha=0.8)

        ax.set_xlabel('Test Case')
        ax.set_ylabel('Wall Time (ms)')
        ax.set_title(f'{test}\nWall Time Comparison')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'wall_time_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'wall_time_comparison.png'}")

def plot_throughput_comparison(df, output_dir):
    """Compare throughput (points/sec) between CPU and GPU."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Filter rows with points_per_sec data
    df_pts = df[df['points_per_sec'] > 0].copy()

    if len(df_pts) == 0:
        print("No throughput data available")
        return

    df_pts['test_label'] = df_pts['test_name'] + '\n' + df_pts['num_points'].astype(str) + ' pts'

    cpu_df = df_pts[df_pts['variant'] == 'CPU']
    gpu_df = df_pts[df_pts['variant'] == 'GPU']

    x = range(len(cpu_df))
    width = 0.35

    ax.bar([i - width/2 for i in x], cpu_df['points_per_sec'] / 1e6, width,
           label='CPU', alpha=0.8, color='steelblue')
    ax.bar([i + width/2 for i in x], gpu_df['points_per_sec'] / 1e6, width,
           label='GPU', alpha=0.8, color='coral')

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Throughput (Million points/sec)')
    ax.set_title('CPU vs GPU Throughput Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(cpu_df['test_label'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'throughput_comparison.png'}")

def plot_speedup(df, output_dir):
    """Calculate and plot GPU speedup over CPU."""
    fig, ax = plt.subplots(figsize=(12, 6))

    speedups = []
    labels = []

    for test in df['test_name'].unique():
        test_df = df[df['test_name'] == test]

        for size in test_df['num_points'].unique():
            size_df = test_df[test_df['num_points'] == size]

            cpu_time = size_df[size_df['variant'] == 'CPU']['wall_time_ms'].values
            gpu_time = size_df[size_df['variant'] == 'GPU']['wall_time_ms'].values

            if len(cpu_time) > 0 and len(gpu_time) > 0:
                speedup = cpu_time[0] / gpu_time[0]
                speedups.append(speedup)

                pts = size if size > 0 else size_df['num_cells'].values[0]
                label = f"{test}\n{pts:,}"
                labels.append(label)

    colors = ['green' if s > 1 else 'red' for s in speedups]
    bars = ax.bar(range(len(speedups)), speedups, color=colors, alpha=0.7)

    ax.axhline(y=1, color='black', linestyle='--', linewidth=1, label='No speedup')
    ax.set_xlabel('Test Case')
    ax.set_ylabel('Speedup (CPU time / GPU time)')
    ax.set_title('GPU Speedup over CPU')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.1f}x',
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'gpu_speedup.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'gpu_speedup.png'}")

def plot_resource_usage(df, output_dir):
    """Plot RAM and VRAM usage."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # RAM usage
    cpu_df = df[df['variant'] == 'CPU']
    gpu_df = df[df['variant'] == 'GPU']

    if len(cpu_df) > 0:
        cpu_df['test_label'] = cpu_df['test_name'] + '\n' + cpu_df['num_points'].astype(str)
        ax1.bar(range(len(cpu_df)), cpu_df['peak_ram_kb'] / 1024, alpha=0.7, label='CPU')
        ax1.set_xlabel('Test Case')
        ax1.set_ylabel('Peak RAM Usage (MB)')
        ax1.set_title('CPU RAM Usage')
        ax1.set_xticks(range(len(cpu_df)))
        ax1.set_xticklabels(cpu_df['test_label'], rotation=45, ha='right', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')

    # VRAM usage
    if len(gpu_df) > 0:
        gpu_df_mem = gpu_df[gpu_df['gpu_mem_used_mb'] > 0].copy()
        if len(gpu_df_mem) > 0:
            gpu_df_mem['test_label'] = gpu_df_mem['test_name'] + '\n' + gpu_df_mem['num_points'].astype(str)
            ax2.bar(range(len(gpu_df_mem)), gpu_df_mem['gpu_mem_used_mb'],
                   alpha=0.7, color='coral', label='GPU')
            ax2.set_xlabel('Test Case')
            ax2.set_ylabel('GPU Memory Used (MB)')
            ax2.set_title('GPU VRAM Usage')
            ax2.set_xticks(range(len(gpu_df_mem)))
            ax2.set_xticklabels(gpu_df_mem['test_label'], rotation=45, ha='right', fontsize=8)
            ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'resource_usage.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'resource_usage.png'}")

def plot_cpu_gpu_utilization(df, output_dir):
    """Plot CPU and GPU utilization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # CPU utilization
    cpu_df = df[(df['variant'] == 'CPU') & (df['cpu_usage_percent'] > 0)].copy()
    if len(cpu_df) > 0:
        cpu_df['test_label'] = cpu_df['test_name'] + '\n' + cpu_df['num_points'].astype(str)
        ax1.bar(range(len(cpu_df)), cpu_df['cpu_usage_percent'], alpha=0.7)
        ax1.axhline(y=100, color='red', linestyle='--', linewidth=1, label='100% (1 core)')
        ax1.set_xlabel('Test Case')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Utilization')
        ax1.set_xticks(range(len(cpu_df)))
        ax1.set_xticklabels(cpu_df['test_label'], rotation=45, ha='right', fontsize=8)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

    # GPU utilization
    gpu_df = df[(df['variant'] == 'GPU') & (df['gpu_utilization_percent'] > 0)].copy()
    if len(gpu_df) > 0:
        gpu_df['test_label'] = gpu_df['test_name'] + '\n' + gpu_df['num_points'].astype(str)
        ax2.bar(range(len(gpu_df)), gpu_df['gpu_utilization_percent'],
               alpha=0.7, color='coral')
        ax2.axhline(y=100, color='red', linestyle='--', linewidth=1, label='100% GPU time')
        ax2.set_xlabel('Test Case')
        ax2.set_ylabel('GPU Utilization (%)')
        ax2.set_title('GPU Utilization (GPU time / Wall time)')
        ax2.set_xticks(range(len(gpu_df)))
        ax2.set_xticklabels(gpu_df['test_label'], rotation=45, ha='right', fontsize=8)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'utilization.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'utilization.png'}")

def generate_summary_table(df, output_dir):
    """Generate summary statistics table."""
    summary = []

    for test in df['test_name'].unique():
        test_df = df[df['test_name'] == test]

        for size in test_df['num_points'].unique():
            size_df = test_df[test_df['num_points'] == size]

            cpu_row = size_df[size_df['variant'] == 'CPU']
            gpu_row = size_df[size_df['variant'] == 'GPU']

            if len(cpu_row) > 0 and len(gpu_row) > 0:
                cpu_time = cpu_row['wall_time_ms'].values[0]
                gpu_time = gpu_row['wall_time_ms'].values[0]
                speedup = cpu_time / gpu_time

                pts = size if size > 0 else size_df['num_cells'].values[0]

                summary.append({
                    'Test': test,
                    'Points/Cells': f"{pts:,}",
                    'CPU Time (ms)': f"{cpu_time:.2f}",
                    'GPU Time (ms)': f"{gpu_time:.2f}",
                    'Speedup': f"{speedup:.2f}x",
                    'CPU RAM (MB)': f"{cpu_row['peak_ram_kb'].values[0] / 1024:.1f}",
                    'GPU VRAM (MB)': f"{gpu_row['gpu_mem_used_mb'].values[0]:.1f}",
                    'GPU Throughput (Mpts/s)': f"{gpu_row['points_per_sec'].values[0] / 1e6:.2f}" if gpu_row['points_per_sec'].values[0] > 0 else "N/A"
                })

    summary_df = pd.DataFrame(summary)

    # Save as CSV
    summary_df.to_csv(output_dir / 'summary_table.csv', index=False)
    print(f"Saved: {output_dir / 'summary_table.csv'}")

    # Print to console
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_performance.py <performance_results.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not Path(csv_file).exists():
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)

    # Load results
    print(f"Loading results from: {csv_file}")
    df = load_results(csv_file)
    print(f"Loaded {len(df)} test results\n")

    # Create output directory
    output_dir = Path('performance_plots')
    output_dir.mkdir(exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'

    # Generate plots
    print("Generating visualizations...")
    plot_wall_time_comparison(df, output_dir)
    plot_throughput_comparison(df, output_dir)
    plot_speedup(df, output_dir)
    plot_resource_usage(df, output_dir)
    plot_cpu_gpu_utilization(df, output_dir)

    # Generate summary table
    generate_summary_table(df, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - wall_time_comparison.png")
    print("  - throughput_comparison.png")
    print("  - gpu_speedup.png")
    print("  - resource_usage.png")
    print("  - utilization.png")
    print("  - summary_table.csv")

if __name__ == '__main__':
    main()
