#!/bin/bash
# PCR GPU Benchmark Docker Runner
set -e

echo "========================================"
echo "PCR GPU Benchmark - Docker Setup"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    echo "Checking NVIDIA Docker runtime..."
    if ! docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo -e "${RED}Error: NVIDIA Docker runtime not found or not working${NC}"
        echo ""
        echo "Please install nvidia-container-toolkit:"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        echo ""
        echo "Quick install (Ubuntu/Debian):"
        echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
        echo "  curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
        echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "  sudo systemctl restart docker"
        exit 1
    fi
    echo -e "${GREEN}✓ NVIDIA Docker runtime is working${NC}"
    echo ""
}

# Check if GPU is available
check_gpu() {
    echo "Checking for NVIDIA GPU..."
    if ! nvidia-smi &>/dev/null; then
        echo -e "${YELLOW}Warning: nvidia-smi not found on host${NC}"
        echo "This might be okay if drivers are only in container"
    else
        echo -e "${GREEN}✓ GPU detected:${NC}"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        echo ""
    fi
}

# Build the Docker image
build_image() {
    echo "Building PCR GPU Docker image..."
    echo "This may take 5-10 minutes on first build..."
    echo ""
    cd ..
    docker build --no-cache -f containers/Dockerfile.cuda -t pcr-gpu-benchmark:latest .

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Docker image built successfully${NC}"
        echo ""
    else
        echo -e "${RED}Error: Docker build failed${NC}"
        exit 1
    fi
}

# Run the benchmark
run_benchmark() {
    echo "Running PCR GPU Benchmark..."
    echo ""

    # Create results directory on host
    mkdir -p ./results

    # Run the container
    docker run --rm --gpus all \
        -v "$(pwd)/results:/workspace/results" \
        pcr-gpu-benchmark:latest

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Benchmark completed successfully${NC}"
        echo ""
        echo "Results are available in: ./results/"
        echo ""
        echo "Files generated:"
        ls -lh ./results/
        echo ""
        echo "To view visualizations:"
        echo "  cd results/performance_plots"
        echo "  open *.png  # (or use your image viewer)"
        echo ""
    else
        echo -e "${RED}Error: Benchmark failed${NC}"
        exit 1
    fi
}

# Interactive shell mode
interactive_shell() {
    echo "Starting interactive shell..."
    echo ""
    echo "Available commands:"
    echo "  ./bench_compare_cpu_gpu     - Run comprehensive benchmark"
    echo "  ./bench_memory              - Memory pool benchmark"
    echo "  ./bench_sort                - Sort benchmark"
    echo "  ./bench_accumulate          - Accumulate benchmark"
    echo "  ./bench_tile_pipeline       - Pipeline benchmark"
    echo "  ctest --output-on-failure   - Run all tests"
    echo ""

    mkdir -p ./results

    docker run -it --rm --gpus all \
        -v "$(pwd)/results:/workspace/results" \
        -w /workspace/build \
        pcr-gpu-benchmark:latest \
        /bin/bash
}

# Run tests
run_tests() {
    echo "Running PCR GPU Tests..."
    echo ""

    mkdir -p ./results

    docker run --rm --gpus all \
        -v "$(pwd)/results:/workspace/results" \
        -w /workspace/build \
        pcr-gpu-benchmark:latest \
        bash -c "ctest --output-on-failure | tee /workspace/results/test_results.txt"

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ All tests passed${NC}"
        echo ""
    else
        echo -e "${YELLOW}Some tests failed. Check ./results/test_results.txt${NC}"
    fi
}

# Run GPU visualization tests
run_visualize() {
    echo "Running GPU visualization tests..."
    echo ""

    # Create output directories on host
    mkdir -p ./pattern_outputs
    mkdir -p ./gpu_pattern_outputs

    docker run --rm --gpus all \
        -v "$(pwd)/pattern_outputs:/workspace/pattern_outputs" \
        -v "$(pwd)/gpu_pattern_outputs:/workspace/gpu_pattern_outputs" \
        -w /workspace \
        pcr-gpu-benchmark:latest \
        bash -c "./run_gpu_visual_tests.sh"

    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Visualization tests completed successfully${NC}"
        echo ""
        echo "Results available in:"
        echo "  - pattern_outputs/ (CPU reference)"
        echo "  - gpu_pattern_outputs/ (GPU output)"
        echo ""
        echo "To view images:"
        echo "  cd pattern_outputs && ls -lh *.png"
        echo "  eog pattern_outputs/*.png  # or your image viewer"
        echo ""
    else
        echo ""
        echo -e "${YELLOW}Some visualization tests failed${NC}"
        echo "Check the output above for details"
    fi
}

# Show usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  build       - Build the Docker image"
    echo "  run         - Run the comprehensive benchmark (default)"
    echo "  test        - Run all unit tests"
    echo "  visualize   - Run GPU visualization tests (CPU vs GPU comparison)"
    echo "  shell       - Start interactive shell in container"
    echo "  check       - Check prerequisites"
    echo "  clean       - Remove Docker image and build artifacts"
    echo ""
    echo "Examples:"
    echo "  $0 build        # Build the image"
    echo "  $0 run          # Run benchmarks"
    echo "  $0 visualize    # Run visualization tests"
    echo "  $0 shell        # Explore interactively"
    echo ""
}

# Clean up
clean() {
    echo "Cleaning up..."
    docker rmi pcr-gpu-benchmark:latest 2>/dev/null || true
    rm -rf ./results ./build
    echo -e "${GREEN}✓ Cleanup complete${NC}"
}

# Main script
main() {
    case "${1:-run}" in
        build)
            check_nvidia_docker
            build_image
            ;;
        run)
            check_nvidia_docker
            check_gpu
            if ! docker images | grep -q "pcr-gpu-benchmark"; then
                echo -e "${YELLOW}Image not found. Building first...${NC}"
                echo ""
                build_image
            fi
            run_benchmark
            ;;
        test)
            check_nvidia_docker
            if ! docker images | grep -q "pcr-gpu-benchmark"; then
                echo -e "${YELLOW}Image not found. Building first...${NC}"
                echo ""
                build_image
            fi
            run_tests
            ;;
        visualize)
            check_nvidia_docker
            if ! docker images | grep -q "pcr-gpu-benchmark"; then
                echo -e "${YELLOW}Image not found. Building first...${NC}"
                echo ""
                build_image
            fi
            run_visualize
            ;;
        shell)
            check_nvidia_docker
            if ! docker images | grep -q "pcr-gpu-benchmark"; then
                echo -e "${YELLOW}Image not found. Building first...${NC}"
                echo ""
                build_image
            fi
            interactive_shell
            ;;
        check)
            check_nvidia_docker
            check_gpu
            echo -e "${GREEN}✓ All prerequisites met${NC}"
            ;;
        clean)
            clean
            ;;
        help|--help|-h)
            usage
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo ""
            usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"
