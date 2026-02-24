#!/bin/bash
# Verification script for PCR development environment

echo "=========================================="
echo "PCR Development Environment Verification"
echo "=========================================="
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_command() {
    local cmd=$1
    local name=$2
    local version_flag=${3:---version}

    if command -v $cmd &> /dev/null; then
        version=$($cmd $version_flag 2>&1 | head -1)
        echo -e "${GREEN}✓${NC} $name: $version"
        return 0
    else
        echo -e "${RED}✗${NC} $name: Not found"
        return 1
    fi
}

check_library() {
    local lib=$1
    local name=$2

    if pkg-config --exists $lib 2>/dev/null; then
        version=$(pkg-config --modversion $lib)
        echo -e "${GREEN}✓${NC} $name: $version"
        return 0
    else
        echo -e "${RED}✗${NC} $name: Not found"
        return 1
    fi
}

check_python_package() {
    local pkg=$1

    if python3 -c "import $pkg" 2>/dev/null; then
        version=$(python3 -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "installed")
        echo -e "${GREEN}✓${NC} Python $pkg: $version"
        return 0
    else
        echo -e "${RED}✗${NC} Python $pkg: Not found"
        return 1
    fi
}

echo "Build Tools:"
check_command gcc "GCC"
check_command g++ "G++"
check_command cmake "CMake"
check_command make "Make" "--version"

echo ""
echo "CUDA Toolkit:"
check_command nvcc "NVCC" "--version"
check_command nvidia-smi "nvidia-smi" "--version"

# Check CUDA headers (CUB and Thrust bundled with CUDA 11+)
if [ -d "/usr/local/cuda/include/cub" ]; then
    echo -e "${GREEN}✓${NC} CUB headers available at /usr/local/cuda/include/cub"
else
    echo -e "${RED}✗${NC} CUB headers not found"
fi

if [ -d "/usr/local/cuda/include/thrust" ]; then
    echo -e "${GREEN}✓${NC} Thrust headers available at /usr/local/cuda/include/thrust"
else
    echo -e "${RED}✗${NC} Thrust headers not found"
fi

# Check GPU access
echo ""
echo "GPU Access:"
if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi >/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        echo -e "${GREEN}✓${NC} GPU detected: $GPU_NAME (${GPU_MEM} MB VRAM)"
    else
        echo -e "${YELLOW}⚠${NC} nvidia-smi present but cannot access GPU (check --gpus all flag)"
    fi
else
    echo -e "${YELLOW}⚠${NC} nvidia-smi not available"
fi

echo ""
echo "Geospatial Libraries:"
check_library gdal "GDAL"
check_library proj "PROJ"

echo ""
echo "Testing Framework:"
if [ -d "/usr/src/googletest" ]; then
    echo -e "${GREEN}✓${NC} GoogleTest: Source available at /usr/src/googletest"
else
    echo -e "${RED}✗${NC} GoogleTest: Not found"
fi

echo ""
echo "Python Packages:"
check_python_package pandas
check_python_package matplotlib
check_python_package seaborn
check_python_package numpy

echo ""
echo "Sudo Access:"
if sudo -n apt-get --version &> /dev/null; then
    echo -e "${GREEN}✓${NC} Passwordless sudo for apt: Available"
else
    echo -e "${RED}✗${NC} Passwordless sudo for apt: Not available"
fi

echo ""
echo "=========================================="
echo "Workspace Permissions:"
echo "=========================================="
ls -ld /workspace
if [ -w /workspace ]; then
    echo -e "${GREEN}✓${NC} /workspace is writable"
else
    echo -e "${RED}✗${NC} /workspace is not writable"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo ""
echo "To build the PCR library:"
echo "  mkdir -p build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release -DPCR_ENABLE_CUDA=ON -DPCR_BUILD_TESTS=ON -DPCR_BUILD_BENCHMARKS=ON"
echo "  make -j\$(nproc)"
echo ""
echo "To run tests:"
echo "  cd build && ctest --output-on-failure"
echo ""
echo "To run benchmarks:"
echo "  cd build && ./bench_compare_cpu_gpu"
echo ""
