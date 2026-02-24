# PCR Library - Repository Rollout & Package Distribution Plan

## Context

The PCR (Point Cloud Reduction) library is a high-performance C++/Python library for rasterizing massive point cloud datasets using CPU, GPU, or hybrid execution modes. The library is feature-complete with comprehensive benchmarks showing GPU acceleration (1.92× speedup), extensive testing (17+ C++ tests, 4 Python tests), and excellent documentation.

**Current State:**
- ✅ Core library fully implemented and tested
- ✅ Python bindings working (pybind11)
- ✅ Comprehensive benchmarks (147.9M points tested)
- ✅ Good documentation (README, API docs, guides)
- ❌ No proper packaging (setup.py is stub)
- ❌ No license file (empty)
- ❌ No CI/CD pipeline
- ❌ No distribution to PyPI/conda

**Goal:**
Transform PCR into a production-ready open source package that:
1. **Gains community adoption** through PyPI/conda distribution
2. **Enables SaaS monetization** via MIT license (open core model)
3. **Automates quality** through CI/CD
4. **Welcomes contributors** with proper guidelines

**Business Model:**
- Open source core library (MIT License)
- Proprietary SaaS/cloud deployment for monetization
- Enterprise support and features as additional revenue

---

## Implementation Plan

### Phase 0: Document the Rollout Plan

**Create `/workspace/ROLLOUT.md`:**
Copy the entire contents of this plan file into a ROLLOUT.md file in the repository. This serves as:
- Project roadmap documentation
- Onboarding guide for contributors
- Reference for implementation checklist
- Historical record of rollout decisions

### Phase 1: Legal Foundation (CRITICAL - Do First)

#### 1.1 Add MIT License

**Create `/workspace/LICENSE`:**
```
MIT License

Copyright (c) 2025 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Delete empty `/workspace/docs/LICENSE`** - convention is root-level LICENSE file

#### 1.2 Add Copyright Headers (Optional but Recommended)

Add to each source file header:
```cpp
// Copyright (c) 2025 [Your Name/Organization]
// SPDX-License-Identifier: MIT
```

---

### Phase 2: Python Packaging (PyPI Distribution)

#### 2.1 Create `pyproject.toml`

**Create `/workspace/pyproject.toml`:**
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel", "pybind11>=2.10.0"]
build-backend = "setuptools.build_meta"

[project]
name = "pcr"
version = "0.1.0"
description = "High-performance point cloud rasterization with CPU/GPU/Hybrid execution modes"
readme = "README.md"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
license = {text = "MIT"}
keywords = ["point-cloud", "lidar", "rasterization", "gpu", "geospatial"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: C++",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: GIS",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
]
las = [
    "laspy>=2.0.0",  # For LAS/LAZ file support
]
viz = [
    "matplotlib>=3.5.0",
    "rasterio>=1.2.0",
]
all = [
    "laspy>=2.0.0",
    "matplotlib>=3.5.0",
    "rasterio>=1.2.0",
]

[project.urls]
Homepage = "https://github.com/yourusername/pcr"
Documentation = "https://github.com/yourusername/pcr#readme"
Repository = "https://github.com/yourusername/pcr"
Issues = "https://github.com/yourusername/pcr/issues"

[tool.setuptools]
packages = ["pcr"]
package-dir = {"" = "python"}

[tool.setuptools.package-data]
pcr = ["*.so", "*.pyd", "*.dll"]
```

#### 2.2 Implement `setup.py`

**Replace `/workspace/python/setup.py`:**
```python
#!/usr/bin/env python3
"""
PCR (Point Cloud Reduction) Python package setup.

This setup.py builds the C++ extension module and installs the Python package.
The actual package metadata is in pyproject.toml.
"""

import os
import sys
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # CMake configuration
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_PYTHON=ON",
            "-DBUILD_TESTS=OFF",  # Don't build tests in package install
        ]

        # Build configuration
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg]

        # Platform-specific settings
        if sys.platform.startswith("darwin"):
            # macOS
            pass
        elif sys.platform.startswith("linux"):
            # Linux
            cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]
        elif sys.platform.startswith("win"):
            # Windows
            cmake_args += [f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"]
            build_args += ["--", "/m"]

        # Build directory
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        # CMake configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=self.build_temp
        )

        # CMake build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args,
            cwd=self.build_temp
        )

setup(
    ext_modules=[CMakeExtension("pcr._pcr", sourcedir="..")],
    cmdclass={"build_ext": CMakeBuild},
)
```

#### 2.3 Create `MANIFEST.in`

**Create `/workspace/MANIFEST.in`:**
```
include LICENSE
include README.md
include pyproject.toml
include CMakeLists.txt

recursive-include include *.h *.hpp
recursive-include src *.cpp *.cu *.h
recursive-include python *.py

exclude .git*
exclude .docker*
exclude build
exclude benchmark_results
exclude dc-lidar
```

#### 2.4 Create Dependencies Files

**Create `/workspace/requirements.txt`:**
```
numpy>=1.20.0
```

**Create `/workspace/requirements-dev.txt`:**
```
-r requirements.txt

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0

# Code quality
black>=22.0.0
flake8>=4.0.0
isort>=5.10.0

# Optional features
laspy>=2.0.0
matplotlib>=3.5.0
rasterio>=1.2.0

# Build tools
pybind11>=2.10.0
cmake>=3.18.0
```

---

### Phase 3: Version Management

#### 3.1 Centralize Version

**Modify `/workspace/python/pcr/__init__.py`:**
```python
"""PCR (Point Cloud Reduction) - High-performance point cloud rasterization."""

__version__ = "0.1.0"

# Read version from file if available
import os
_version_file = os.path.join(os.path.dirname(__file__), "VERSION")
if os.path.exists(_version_file):
    with open(_version_file) as f:
        __version__ = f.read().strip()

# Import core extension
from . import _pcr
# ... rest of imports
```

**Create `/workspace/VERSION`:**
```
0.1.0
```

**Update `/workspace/CMakeLists.txt` to read VERSION:**
```cmake
# Read version from file
file(READ "${CMAKE_SOURCE_DIR}/VERSION" PCR_VERSION)
string(STRIP "${PCR_VERSION}" PCR_VERSION)
project(pcr VERSION ${PCR_VERSION} LANGUAGES CXX)
```

---

### Phase 4: CI/CD Pipeline (GitHub Actions)

#### 4.1 Build & Test Workflow

**Create `.github/workflows/build-test.yml`:**
```yaml
name: Build and Test

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  build-linux:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y \
          cmake \
          libgdal-dev \
          libproj-dev \
          libomp-dev

    - name: Install Python dependencies
      run: |
        pip install -r requirements-dev.txt

    - name: Build C++ library
      run: |
        mkdir build
        cd build
        cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON=ON
        cmake --build . -j$(nproc)

    - name: Run C++ tests
      run: |
        cd build
        ctest --output-on-failure -j$(nproc)

    - name: Build Python package
      run: |
        cd python
        pip install -e .

    - name: Run Python tests
      run: |
        pytest tests/python -v --cov=pcr --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        flags: python-${{ matrix.python-version }}

  build-macos:
    runs-on: macos-latest

    steps:
    - uses: actions/checkout@v3

    - name: Install dependencies
      run: |
        brew install cmake gdal proj libomp

    - name: Build and test
      run: |
        mkdir build && cd build
        cmake .. -DBUILD_TESTS=ON
        cmake --build . -j$(sysctl -n hw.ncpu)
        ctest --output-on-failure
```

#### 4.2 Lint Workflow

**Create `.github/workflows/lint.yml`:**
```yaml
name: Code Quality

on: [push, pull_request]

jobs:
  python-lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    - name: Install linters
      run: pip install black flake8 isort
    - name: Check formatting (black)
      run: black --check python/ tests/python/
    - name: Check imports (isort)
      run: isort --check python/ tests/python/
    - name: Lint (flake8)
      run: flake8 python/ tests/python/ --max-line-length=100

  cpp-format:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Check C++ formatting
      run: |
        find src include -name '*.cpp' -o -name '*.h' -o -name '*.cu' | \
          xargs clang-format --dry-run --Werror
```

#### 4.3 Release Workflow

**Create `.github/workflows/release.yml`:**
```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build-wheels:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ["3.8", "3.9", "3.10", "3.11"]

    steps:
    - uses: actions/checkout@v3

    - uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Build wheel
      run: |
        pip install build
        python -m build --wheel

    - name: Upload wheel
      uses: actions/upload-artifact@v3
      with:
        name: wheels
        path: dist/*.whl

  publish-pypi:
    needs: build-wheels
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')

    steps:
    - uses: actions/download-artifact@v3
      with:
        name: wheels
        path: dist/

    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

---

### Phase 5: Documentation & Community

#### 5.1 Create CONTRIBUTING.md

**Create `/workspace/CONTRIBUTING.md`:**
```markdown
# Contributing to PCR

Thank you for considering contributing to PCR! This document provides guidelines for contributing.

## Development Setup

### Prerequisites
- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- Python 3.8+
- GDAL and PROJ libraries
- (Optional) CUDA Toolkit 11.0+ for GPU support

### Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/pcr.git
   cd pcr
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

3. Build C++ library:
   ```bash
   mkdir build && cd build
   cmake .. -DBUILD_TESTS=ON -DBUILD_PYTHON=ON
   cmake --build . -j$(nproc)
   ```

4. Run tests:
   ```bash
   ctest --output-on-failure
   cd ../python && pip install -e .
   pytest ../tests/python
   ```

## Code Style

- **C++**: Follow Google C++ Style Guide, use `clang-format`
- **Python**: Follow PEP 8, use `black` formatter
- **Commits**: Use conventional commits (feat:, fix:, docs:, etc.)

## Testing

All new features must include:
- C++ unit tests (GoogleTest)
- Python tests (pytest)
- Documentation updates

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
```

#### 5.2 Create CHANGELOG.md

**Create `/workspace/CHANGELOG.md`:**
```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-02-22

### Added
- Initial public release
- Core point cloud rasterization engine
- CPU, GPU, and Hybrid execution modes
- Python bindings via pybind11
- Comprehensive test suite
- Benchmark scripts and DC LiDAR validation
- Documentation and examples

### Performance
- GPU acceleration: 1.92× speedup over CPU multi-threaded
- Throughput: 10.9 million points/second on NVIDIA RTX 2060
- Tested with 147.9M points, 62.8M grid cells

[0.1.0]: https://github.com/yourusername/pcr/releases/tag/v0.1.0
```

#### 5.3 Update README.md

**Add to `/workspace/README.md`:**
```markdown
## Installation

### From PyPI (Recommended)

```bash
pip install pcr
```

### From Source

See [CONTRIBUTING.md](CONTRIBUTING.md) for build instructions.

## Quick Start

```python
import pcr
import numpy as np

# Create a simple point cloud
points = pcr.PointCloud.create(1000)
points.set_x_array(np.random.rand(1000) * 100)
points.set_y_array(np.random.rand(1000) * 100)
points.add_channel('elevation', pcr.DataType.Float32)
points.set_channel_array_f32('elevation', np.random.rand(1000) * 50)

# Configure grid
gc = pcr.GridConfig()
gc.bounds.min_x, gc.bounds.min_y = 0, 0
gc.bounds.max_x, gc.bounds.max_y = 100, 100
gc.cell_size_x = gc.cell_size_y = 1.0
gc.compute_dimensions()

# Create reduction spec
spec = pcr.ReductionSpec()
spec.value_channel = 'elevation'
spec.type = pcr.ReductionType.Average

# Run pipeline
cfg = pcr.PipelineConfig()
cfg.grid = gc
cfg.reductions = [spec]
cfg.exec_mode = pcr.ExecutionMode.GPU  # or CPU, Hybrid
cfg.output_path = 'output.tif'

pipe = pcr.Pipeline.create(cfg)
pipe.ingest(points)
pipe.finalize()

# Access results
result = pipe.result()
elevation_grid = result.band_array(0)
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
```

---

### Phase 6: Conda Distribution (Optional, can do later)

#### 6.1 Create Conda Recipe

**Create `/workspace/conda/meta.yaml`:**
```yaml
{% set name = "pcr" %}
{% set version = "0.1.0" %}

package:
  name: {{ name|lower }}
  version: {{ version }}

source:
  path: ..

build:
  number: 0
  skip: true  # [py<38]

requirements:
  build:
    - {{ compiler('cxx') }}
    - cmake >=3.18
    - pybind11 >=2.10
  host:
    - python
    - numpy
    - gdal
    - proj
  run:
    - python
    - numpy
    - gdal
    - proj

test:
  imports:
    - pcr
  commands:
    - pytest tests/python

about:
  home: https://github.com/yourusername/pcr
  license: MIT
  license_file: LICENSE
  summary: High-performance point cloud rasterization
  description: |
    PCR provides GPU-accelerated rasterization of massive point cloud
    datasets with CPU, GPU, and hybrid execution modes.
```

---

### Phase 7: Docker Distribution

#### 7.1 Optimize Dockerfile

**Update `/workspace/Dockerfile` (new file):**
```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    libgdal-dev \
    libproj-dev \
    libomp-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Build PCR
WORKDIR /build
COPY . .
RUN mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON && \
    cmake --build . -j$(nproc) && \
    cmake --install .

# Runtime image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libgdal30 \
    libproj22 \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy built library
COPY --from=builder /usr/local /usr/local
COPY --from=builder /build/python/pcr /usr/local/lib/python3.10/dist-packages/pcr

# Install Python dependencies
RUN pip3 install numpy

WORKDIR /workspace
CMD ["python3"]
```

#### 7.2 Docker Compose for Development

**Create `/workspace/docker-compose.yml`:**
```yaml
version: '3.8'

services:
  pcr-dev:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - .:/workspace
      - ./data:/data
    environment:
      - CUDA_VISIBLE_DEVICES=0
    runtime: nvidia
```

---

## Critical Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `/workspace/LICENSE` | **Create** | MIT License text (CRITICAL) |
| `/workspace/ROLLOUT.md` | **Create** | This complete rollout plan as repository documentation |
| `/workspace/pyproject.toml` | **Create** | Python package metadata |
| `/workspace/python/setup.py` | **Replace** | CMake-based build system |
| `/workspace/MANIFEST.in` | **Create** | Package file inclusion rules |
| `/workspace/requirements.txt` | **Create** | Runtime dependencies |
| `/workspace/requirements-dev.txt` | **Create** | Development dependencies |
| `/workspace/VERSION` | **Create** | Centralized version number |
| `/workspace/CONTRIBUTING.md` | **Create** | Contributor guidelines |
| `/workspace/CHANGELOG.md` | **Create** | Version history |
| `/workspace/.github/workflows/build-test.yml` | **Create** | CI/CD build pipeline |
| `/workspace/.github/workflows/lint.yml` | **Create** | Code quality checks |
| `/workspace/.github/workflows/release.yml` | **Create** | Automated releases |
| `/workspace/Dockerfile` | **Create/Update** | Production Docker image |
| `/workspace/README.md` | **Update** | Add installation & quick start |
| `/workspace/python/pcr/__init__.py` | **Update** | Version from file |
| `/workspace/CMakeLists.txt` | **Update** | Read VERSION file |

---

## Verification & Testing Plan

### 1. Local Package Build
```bash
# Build source distribution
python -m build --sdist

# Build wheel
python -m build --wheel

# Install locally
pip install dist/pcr-0.1.0-*.whl

# Test import
python -c "import pcr; print(pcr.__version__)"
```

### 2. Test PyPI Upload (Dry Run)
```bash
# Upload to Test PyPI first
pip install twine
twine upload --repository testpypi dist/*

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ pcr

# Verify it works
python -c "import pcr; print(pcr.__version__)"
```

### 3. CI/CD Verification
```bash
# Push to GitHub and verify workflows run
git add .
git commit -m "feat: add packaging infrastructure"
git push

# Check GitHub Actions tab for green checkmarks
```

### 4. Docker Build
```bash
# Build Docker image
docker build -t pcr:latest .

# Run tests in Docker
docker run pcr:latest python3 -c "import pcr; print(pcr.__version__)"
```

---

## Release Checklist

Before first public release (v0.1.0):

- [ ] LICENSE file created with MIT text
- [ ] pyproject.toml with complete metadata
- [ ] setup.py implemented with CMake build
- [ ] requirements.txt with dependencies
- [ ] CONTRIBUTING.md written
- [ ] CHANGELOG.md started
- [ ] README.md updated with installation
- [ ] CI/CD workflows passing
- [ ] All tests passing locally
- [ ] Package builds successfully
- [ ] Test PyPI upload works
- [ ] Documentation reviewed
- [ ] Version tagged in git

---

## Phased Rollout Timeline

### Week 1: Foundation
- Day 1-2: Add LICENSE, create packaging files (pyproject.toml, setup.py)
- Day 3-4: Set up CI/CD (GitHub Actions workflows)
- Day 5: Test builds locally, fix issues

### Week 2: Quality & Documentation
- Day 6-7: Add CONTRIBUTING.md, update README
- Day 8-9: Create code quality tools (formatters, linters)
- Day 10: Documentation review and polish

### Week 3: Distribution
- Day 11-12: Test PyPI upload (testpypi)
- Day 13: Production PyPI release
- Day 14: Docker image optimization and publish
- Day 15: Conda recipe creation

### Week 4: Community Launch
- Day 16-17: Write blog post, create examples
- Day 18-19: Social media announcement, HN/Reddit post
- Day 20: Monitor issues, respond to community

---

## Success Criteria

- ✅ MIT License in place
- ✅ Package installable via `pip install pcr`
- ✅ CI/CD pipeline green on all PRs
- ✅ Documentation complete and accessible
- ✅ 100+ stars on GitHub (community interest)
- ✅ 5+ contributors (community engagement)
- ✅ Docker image available and functional
- ✅ Conda package available (optional)
