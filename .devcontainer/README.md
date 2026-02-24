# PCR Development Container

This devcontainer is configured for C++/CUDA development with the PCR (Point Cloud Raster) library.

## Features

### Development Tools
- **C++ Build Tools**: GCC 11+, CMake 3.22+, build-essential
- **CUDA Toolkit**: NVIDIA CUDA 13.0 with nvcc compiler, CUB, and Thrust libraries
- **GPU Access**: Enabled via `--gpus all` runtime flag
- **Geospatial Libraries**: GDAL 3.4, PROJ 8.2
- **Testing Framework**: GoogleTest
- **Python Tools**: Python 3.10, pandas, matplotlib, seaborn, numpy

### VS Code Extensions
- **C/C++ IntelliSense**: `ms-vscode.cpptools`
- **CMake Tools**: `ms-vscode.cmake-tools`, `twxs.cmake`
- **NVIDIA Nsight**: `nvidia.nsight-vscode-edition` (GPU debugging)
- **Claude Code**: `anthropic.claude-code`
- **GitLens**: `eamodio.gitlens`

### Sudo Access
The `node` user has passwordless sudo for:
- Package management: `apt`, `apt-get`, `dpkg`, `pip3`
- Firewall configuration: `/usr/local/bin/init-firewall.sh`

## Getting Started

### 1. Rebuild the Container

After updating the devcontainer configuration, rebuild:

**VS Code:**
- Press `Cmd/Ctrl + Shift + P`
- Select: `Dev Containers: Rebuild Container`

**Command Line:**
```bash
# From outside the container
docker compose -f .devcontainer/docker-compose.yml up --build
```

### 2. Verify Setup

Run the verification script to check all dependencies:

```bash
./.devcontainer/verify-setup.sh
```

### 3. Build PCR Library

```bash
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPCR_ENABLE_CUDA=ON \
  -DPCR_BUILD_TESTS=ON \
  -DPCR_BUILD_BENCHMARKS=ON
make -j$(nproc)
```

### 4. Run Tests

```bash
cd build
ctest --output-on-failure
```

### 5. Run Benchmarks

```bash
cd build
./bench_compare_cpu_gpu
```

## Installing Additional Packages

You can now install packages without password prompts:

```bash
# Install additional apt packages
sudo apt-get update
sudo apt-get install <package-name>

# Install Python packages
sudo pip3 install <package-name>
```

## CUDA and GPU Support

This devcontainer includes the **NVIDIA CUDA 13.0 toolkit** with full GPU development support.

### What's Included
- **CUDA Compiler**: nvcc for compiling .cu files
- **CUDA Libraries**: CUB, Thrust (bundled with CUDA toolkit)
- **GPU Debugging**: NVIDIA Nsight VS Code extension pre-installed
- **GPU Runtime**: nvidia-smi for monitoring GPU utilization

### Verify GPU Access

Check that GPU is accessible:
```bash
nvidia-smi          # Should show GPU name and memory
nvcc --version      # Should show CUDA 13.0
```

### Building PCR with GPU Support

Build the library with CUDA enabled:
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPCR_ENABLE_CUDA=ON -DPCR_BUILD_TESTS=ON
make -j$(nproc)
ctest --output-on-failure
```

### Troubleshooting

**"Could not select device driver"**
- Ensure NVIDIA Container Toolkit is installed on host
- Check devcontainer.json has `"--gpus", "all"` in runArgs

**"Driver/library version mismatch"**
- Reboot host machine after driver update
- Or reload drivers: `sudo rmmod nvidia_uvm && sudo modprobe nvidia`

**No GPU detected in container**
- Verify on host: `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi`
- Check Docker daemon has nvidia runtime configured

## Troubleshooting

### "Permission denied" when installing packages

Make sure you've rebuilt the container after updating the Dockerfile:
```bash
# In VS Code
Cmd/Ctrl + Shift + P → "Dev Containers: Rebuild Container"
```

### GPU not detected

1. Check host GPU: `nvidia-smi` (outside container)
2. Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi`
3. If both work but devcontainer fails, check `runArgs` includes `"--gpus", "all"`
4. Rebuild the container: Cmd/Ctrl + Shift + P → "Dev Containers: Rebuild Container"

### CMake configuration fails

Check all dependencies are installed:
```bash
./.devcontainer/verify-setup.sh
```

Missing dependencies can be installed with:
```bash
sudo apt-get install <missing-package>
```

## File Structure

```
.devcontainer/
├── Dockerfile              # Container image definition
├── devcontainer.json       # VS Code devcontainer config
├── init-firewall.sh        # Firewall setup script
├── verify-setup.sh         # Dependency verification script
└── README.md              # This file
```

## Environment Variables

- `DEVCONTAINER=true` - Indicates running in devcontainer
- `NODE_OPTIONS=--max-old-space-size=4096` - Node.js memory limit
- `CLAUDE_CONFIG_DIR=/home/node/.claude` - Claude Code config
- `SHELL=/bin/zsh` - Default shell
- `EDITOR=nano` - Default editor

## Volumes

- **Command history**: `claude-code-bashhistory-${devcontainerId}`
- **Claude config**: `claude-code-config-${devcontainerId}`
- **Workspace**: Bind mount from host `${localWorkspaceFolder}` to `/workspace`

## Next Steps

1. Run `./.devcontainer/verify-setup.sh` to verify CUDA and all dependencies
2. Build the library with CUDA: `mkdir build && cd build && cmake .. -DPCR_ENABLE_CUDA=ON && make -j$(nproc)`
3. Run tests: `cd build && ctest --output-on-failure`
4. Run GPU benchmarks: `cd build && ./bench_compare_cpu_gpu`
5. Monitor GPU during execution: `watch -n 1 nvidia-smi`

The dev container now provides the full CUDA development environment. You can develop, compile, debug, and benchmark GPU code directly in VS Code with full IntelliSense support for CUDA kernels.

## Support

- See `DOCKER_QUICKSTART.md` for Docker usage
- See `BENCHMARKING_QUICKSTART.md` for benchmark details
- See `CLAUDE.md` for project architecture
