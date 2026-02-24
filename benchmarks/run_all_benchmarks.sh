#!/bin/bash
set -e

echo "========================================"
echo "PCR Performance Benchmark Suite"
echo "========================================"
echo ""

# Check if we're in the build directory
if [ ! -f "./bench_compare_cpu_gpu" ]; then
    echo "Error: Benchmarks not found in current directory"
    echo "Please run this script from the build directory:"
    echo "  cd build && ../benchmarks/run_all_benchmarks.sh"
    exit 1
fi

# Create output directory
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo ""

# System info
echo "=== System Information ==="
echo "CPU: $(lscpu | grep 'Model name' | sed 's/Model name:\s*//')"
echo "RAM: $(free -h | grep Mem | awk '{print $2}')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
fi
echo ""

# Run individual benchmarks
echo "=== Running Individual Benchmarks ==="

echo "1. Memory Pool Benchmark..."
./bench_memory > "$OUTPUT_DIR/bench_memory.txt" 2>&1
echo "   Done. Output: $OUTPUT_DIR/bench_memory.txt"

echo "2. Sort Benchmark..."
./bench_sort > "$OUTPUT_DIR/bench_sort.txt" 2>&1
echo "   Done. Output: $OUTPUT_DIR/bench_sort.txt"

echo "3. Accumulate Benchmark..."
./bench_accumulate > "$OUTPUT_DIR/bench_accumulate.txt" 2>&1
echo "   Done. Output: $OUTPUT_DIR/bench_accumulate.txt"

echo "4. Pipeline Benchmark..."
./bench_tile_pipeline > "$OUTPUT_DIR/bench_tile_pipeline.txt" 2>&1
echo "   Done. Output: $OUTPUT_DIR/bench_tile_pipeline.txt"

echo ""
echo "=== Running Comprehensive CPU vs GPU Comparison ==="
./bench_compare_cpu_gpu | tee "$OUTPUT_DIR/bench_compare_cpu_gpu.txt"

# Move CSV to output directory
if [ -f "performance_results.csv" ]; then
    mv performance_results.csv "$OUTPUT_DIR/"
    echo ""
    echo "Performance data saved to: $OUTPUT_DIR/performance_results.csv"
fi

# Generate visualizations if Python is available
if command -v python3 &> /dev/null; then
    echo ""
    echo "=== Generating Visualizations ==="

    # Check for required Python packages
    python3 -c "import pandas, matplotlib, seaborn" 2>/dev/null
    if [ $? -eq 0 ]; then
        cd "$OUTPUT_DIR"
        python3 ../../benchmarks/visualize_performance.py performance_results.csv
        cd ..

        echo ""
        echo "Visualizations saved to: $OUTPUT_DIR/performance_plots/"
    else
        echo "Warning: Python packages (pandas, matplotlib, seaborn) not found"
        echo "Install with: pip install pandas matplotlib seaborn"
        echo "Then run manually:"
        echo "  python3 ../benchmarks/visualize_performance.py $OUTPUT_DIR/performance_results.csv"
    fi
else
    echo ""
    echo "Python3 not found. Skipping visualization generation."
fi

echo ""
echo "========================================"
echo "Benchmark Suite Complete!"
echo "========================================"
echo ""
echo "Results location: $OUTPUT_DIR/"
echo ""
echo "Files generated:"
ls -lh "$OUTPUT_DIR/" | tail -n +2
echo ""
echo "To view visualizations:"
echo "  cd $OUTPUT_DIR/performance_plots"
echo "  # Open PNG files with image viewer"
echo ""
