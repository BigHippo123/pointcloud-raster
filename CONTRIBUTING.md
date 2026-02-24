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
   git clone https://github.com/pcr-dev/pcr.git
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
