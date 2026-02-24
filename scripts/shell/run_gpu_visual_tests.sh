#!/bin/bash
# GPU Visual Validation Test Suite
# Generates patterns on CPU and GPU, then compares outputs
set -e

export LINE_PROFILE=1

# Ensure PYTHONPATH is set for pcr module
export PYTHONPATH=/workspace/python:${PYTHONPATH}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo "========================================"
echo "GPU Visual Validation Test Suite"
echo "========================================"
echo ""

# Check if CUDA is available
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        echo ""
    else
        echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
        echo ""
    fi
}

# Step 0: Check CUDA availability
echo "Step 0: Checking GPU availability..."
check_cuda

# Step 1: Generate CPU reference patterns
echo "========================================"
echo "Step 1: Generating CPU reference patterns..."
echo "========================================"
echo ""

if [ ! -f "generate_all_patterns.py" ]; then
    echo "Error: generate_all_patterns.py not found"
    exit 1
fi

python3 generate_all_patterns.py
if [ $? -ne 0 ]; then
    echo "Error: CPU pattern generation failed"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ CPU patterns generated${NC}"
echo ""

# Step 2: Generate GPU patterns
echo "========================================"
echo "Step 2: Generating GPU patterns..."
echo "========================================"
echo ""

if [ ! -f "generate_gpu_patterns.py" ]; then
    echo "Error: generate_gpu_patterns.py not found"
    exit 1
fi

python3 generate_gpu_patterns.py
if [ $? -ne 0 ]; then
    echo "Error: GPU pattern generation failed"
    exit 1
fi

echo ""
echo -e "${GREEN}✓ GPU patterns generated${NC}"
echo ""

# Step 3: Compare outputs
echo "========================================"
echo "Step 3: Comparing CPU vs GPU outputs..."
echo "========================================"
echo ""

if [ ! -f "compare_cpu_gpu_patterns.py" ]; then
    echo "Error: compare_cpu_gpu_patterns.py not found"
    exit 1
fi

python3 compare_cpu_gpu_patterns.py
COMPARE_RESULT=$?

echo ""

# Final summary
echo "========================================"
echo "Test Suite Complete!"
echo "========================================"
echo ""

if [ $COMPARE_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests PASSED${NC}"
    echo ""
    echo "Output directories:"
    echo "  - pattern_outputs/     (CPU reference)"
    echo "  - gpu_pattern_outputs/ (GPU output)"
    echo ""
    echo "To view images:"
    echo "  eog pattern_outputs/*.png gpu_pattern_outputs/*.png"
    exit 0
else
    echo -e "${YELLOW}Some tests FAILED or had errors${NC}"
    echo ""
    echo "Review the comparison output above for details."
    echo ""
    echo "Output directories:"
    echo "  - pattern_outputs/     (CPU reference)"
    echo "  - gpu_pattern_outputs/ (GPU output)"
    echo ""
    echo "To manually compare specific patterns:"
    echo "  python3 compare_cpu_gpu_patterns.py --pattern 01_checkerboard"
    exit 1
fi
