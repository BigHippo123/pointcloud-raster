#!/usr/bin/env bash
# Run all PCR Python benchmarks and generate a Markdown + HTML report.
#
# Usage:
#   scripts/shell/run_benchmarks.sh [--quick] [--no-gpu] [--billion]
#
#   --quick    Skip the full glyph benchmark (bench_glyphs.py only, ~60s)
#   --no-gpu   Force CPU-only mode (useful for profiling or headless machines)
#   --billion  Also run the billion-point scale test (slow, needs lots of RAM)
#
# Output:
#   benchmark_results/<TIMESTAMP>/
#     system_info.txt          system metadata (key=value)
#     cpu_vs_gpu.csv           CPU vs GPU throughput data
#     multithread.csv          thread-scaling data
#     hybrid.csv               hybrid mode data
#     glyphs_full.csv          glyph throughput data (unless --quick)
#     bench_glyphs_chart.png   quick glyph chart
#     glyphs_full_chart.png    full glyph chart (unless --quick)
#     *.txt                    raw stdout from each benchmark
#     benchmark_report.md      GitHub-flavored Markdown report
#     benchmark_report.html    Standalone HTML report
#
#   benchmark_results/latest -> <TIMESTAMP>  (symlink updated after each run)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/../.." && pwd)"
BENCH_DIR="$WORKSPACE/scripts/benchmarks"
RESULTS_ROOT="$WORKSPACE/benchmark_results"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
RESULTS_DIR="$RESULTS_ROOT/$TIMESTAMP"

export PYTHONPATH="$WORKSPACE/python"

# ── Parse flags ───────────────────────────────────────────────────────────────
QUICK=false
NO_GPU=false
BILLION=false

for arg in "$@"; do
    case "$arg" in
        --quick)   QUICK=true ;;
        --no-gpu)  NO_GPU=true ;;
        --billion) BILLION=true ;;
        *)
            echo "Unknown flag: $arg"
            echo "Usage: $0 [--quick] [--no-gpu] [--billion]"
            exit 1
            ;;
    esac
done

# Validate that PCR library can be imported before starting
if ! python3 -c "import pcr" 2>/dev/null; then
    echo "ERROR: Cannot import pcr. Make sure the library is built."
    echo "  cmake --build $WORKSPACE/build --target _pcr -j\$(nproc)"
    exit 1
fi

# ── Detect GPU ────────────────────────────────────────────────────────────────
HAS_GPU=false
if ! $NO_GPU && command -v nvidia-smi &>/dev/null; then
    if nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null 2>&1; then
        HAS_GPU=true
    fi
fi

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              PCR Python Benchmark Suite                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Results  : $RESULTS_DIR"
echo "  PYTHONPATH: $PYTHONPATH"
$HAS_GPU && echo "  GPU      : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
$HAS_GPU || echo "  GPU      : not available (CPU-only run)"
$QUICK   && echo "  Mode     : quick (skipping full glyph benchmark)"
echo ""

# ── Collect system information ────────────────────────────────────────────────
{
    echo "timestamp=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "hostname=$(hostname 2>/dev/null || echo unknown)"

    if [ -f /proc/cpuinfo ]; then
        echo "cpu=$(grep 'model name' /proc/cpuinfo | head -1 | sed 's/model name\s*:\s*//')"
        echo "cpu_cores=$(nproc)"
    fi

    if [ -f /proc/meminfo ]; then
        echo "ram_gb=$(awk '/MemTotal/ {printf "%.1f", $2/1024/1024}' /proc/meminfo)"
    fi

    if $HAS_GPU; then
        echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
        echo "gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1) MiB"
        # CUDA version from nvidia-smi header
        cuda=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' | head -1 || true)
        [ -n "$cuda" ] && echo "cuda_version=$cuda"
    else
        echo "gpu=none"
    fi

    python3 -c "import pcr; print('pcr_version=' + pcr.__version__)" 2>/dev/null || \
        echo "pcr_version=unknown"
} > "$RESULTS_DIR/system_info.txt"

echo "System info:"
sed 's/^/  /' "$RESULTS_DIR/system_info.txt"
echo ""

# ── Benchmark runner helper ───────────────────────────────────────────────────
# Runs a Python benchmark, captures stdout, continues on failure.
run_bench() {
    local label="$1"
    local script="$2"
    shift 2
    local out="$RESULTS_DIR/${label}.txt"
    printf "  %-38s" "$label ..."
    if python3 "$script" "$@" > "$out" 2>&1; then
        echo " ✓"
    else
        echo " ✗  (see $(basename "$out"))"
    fi
}

# Copies a CSV from its hardcoded scripts/ output path to the results dir.
copy_csv() {
    local src="$1"
    local dest="$2"
    [ -f "$WORKSPACE/scripts/$src" ] && cp "$WORKSPACE/scripts/$src" "$RESULTS_DIR/$dest" || true
}

copy_png() {
    local src="$1"
    local dest="$2"
    [ -f "$WORKSPACE/scripts/$src" ] && cp "$WORKSPACE/scripts/$src" "$RESULTS_DIR/$dest" || true
}

# ── Run benchmarks ────────────────────────────────────────────────────────────
echo "Running benchmarks..."
echo ""

# 1. CPU multi-thread scaling (always runs; adds GPU row if available)
run_bench "multithread" \
    "$BENCH_DIR/benchmark_multithread.py"
copy_csv "multithread_benchmark_results.csv" "multithread.csv"

# 2. CPU vs GPU (skips gracefully if no GPU)
run_bench "cpu_vs_gpu" \
    "$BENCH_DIR/benchmark_cpu_gpu.py"
copy_csv "performance_comparison.csv" "cpu_vs_gpu.csv"

# 3. Hybrid mode (CPU-MT baseline; adds GPU/Hybrid rows if available)
run_bench "hybrid" \
    "$BENCH_DIR/benchmark_hybrid.py"
copy_csv "hybrid_benchmark_results.csv" "hybrid.csv"

# 4. Quick glyph benchmark (~60s, always runs)
run_bench "bench_glyphs" \
    "$BENCH_DIR/bench_glyphs.py"
copy_png "bench_glyphs_chart.png" "bench_glyphs_chart.png"

# 5. Full glyph benchmark (skipped with --quick)
if ! $QUICK; then
    GLYPH_MODE="both"
    $HAS_GPU || GLYPH_MODE="cpu"
    run_bench "glyphs_full" \
        "$BENCH_DIR/benchmark_glyph_full.py" \
        --mode "$GLYPH_MODE" --repeats 2
    copy_csv "benchmark_glyph_results.csv" "glyphs_full.csv"
    copy_png "benchmark_glyph_chart.png" "glyphs_full_chart.png"
fi

# 6. Glyph visual gallery (generate_glyph_patterns.py → 8 PNGs)
if ! $QUICK; then
    PATTERN_MODE="cpu"
    $HAS_GPU && PATTERN_MODE="gpu"
    printf "  %-38s" "glyph_patterns ..."
    GLYPH_OUT="$WORKSPACE/glyph_pattern_outputs"
    if python3 "$WORKSPACE/scripts/patterns/generate_glyph_patterns.py" \
            --mode "$PATTERN_MODE" > "$RESULTS_DIR/glyph_patterns.txt" 2>&1; then
        echo " ✓"
        # Copy the 8 gallery PNGs into results dir with "glyph_" prefix
        for png in "$GLYPH_OUT"/0[1-8]_*.png; do
            [ -f "$png" ] && cp "$png" "$RESULTS_DIR/glyph_$(basename "$png")" || true
        done
    else
        echo " ✗  (see glyph_patterns.txt)"
    fi
fi

# 7. DC LiDAR real-world example (optional — needs las/laz files)
DC_LAS_DIR="${DC_LAS_DIR:-$WORKSPACE/dc-lidar}"
if ! $QUICK && [ -d "$DC_LAS_DIR" ] && [ "$(ls -A "$DC_LAS_DIR" 2>/dev/null)" ]; then
    DC_GLYPH_OUT="$WORKSPACE/glyph_pattern_outputs"
    DC_MODE="cpu-mt"
    $HAS_GPU && DC_MODE="gpu"
    printf "  %-38s" "dc_lidar ..."
    if python3 "$WORKSPACE/scripts/data/test_dc_lidar.py" \
            --las-dir "$DC_LAS_DIR" \
            --outdir  "$DC_GLYPH_OUT" \
            --mode    "$DC_MODE" \
            --glyph   all \
            --subset  5 \
            --csv     "$RESULTS_DIR/dc_lidar.csv" \
            > "$RESULTS_DIR/dc_lidar.txt" 2>&1; then
        echo " ✓"
        # Render TIF outputs into a comparison PNG
        printf "  %-38s" "render_dc_tifs ..."
        if python3 "$BENCH_DIR/render_dc_tifs.py" \
                --tif-dir "$DC_GLYPH_OUT" \
                --mode    "$DC_MODE" \
                --output  "$RESULTS_DIR/dc_comparison.png" \
                >> "$RESULTS_DIR/dc_lidar.txt" 2>&1; then
            echo " ✓"
        else
            echo " ✗"
        fi
    else
        echo " ✗  (see dc_lidar.txt)"
    fi
elif ! $QUICK; then
    echo "  dc_lidar ...                           (skipped — no LAS files at $DC_LAS_DIR)"
fi

# 8. Billion-point scale test (opt-in only)
if $BILLION; then
    echo ""
    POINTS="${BILLION_POINTS:-10000000}"
    echo "  Running billion-point test with $POINTS points..."
    run_bench "billion_points" \
        "$BENCH_DIR/benchmark_billion_points.py" \
        --points "$POINTS"
    copy_csv "benchmark_results.csv" "billion_points.csv"
fi

# ── Generate report ───────────────────────────────────────────────────────────
echo ""
echo "Generating report..."
python3 "$BENCH_DIR/generate_report.py" --results-dir "$RESULTS_DIR"

# ── Update 'latest' symlink ───────────────────────────────────────────────────
LATEST="$RESULTS_ROOT/latest"
rm -f "$LATEST"
ln -sf "$TIMESTAMP" "$LATEST"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Done!                                                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Results  : $RESULTS_DIR"
echo "  Markdown : $RESULTS_DIR/benchmark_report.md"
echo "  HTML     : $RESULTS_DIR/benchmark_report.html"
echo "  Latest   : $RESULTS_ROOT/latest/"
echo ""
echo "  View in browser:"
echo "    open $RESULTS_DIR/benchmark_report.html"
echo ""
echo "  Contents:"
ls -lh "$RESULTS_DIR/" | awk 'NR>1 {printf "    %s\n", $0}'
echo ""
