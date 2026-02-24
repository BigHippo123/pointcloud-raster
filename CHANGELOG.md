# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-22

### Added
- Initial public release
- Core point cloud rasterization engine
- CPU, GPU, and Hybrid execution modes
- Python bindings via pybind11
- Comprehensive test suite (17 C++ tests, 4 Python tests)
- Benchmark scripts and DC LiDAR validation
- Documentation and examples
- Glyph-based splatting (Gaussian and Line glyphs)
- Multiple reduction types (Average, Min, Max, Weighted Average, etc.)
- GeoTIFF I/O support with GDAL
- Point cloud format support (XYZ, CSV, binary)

### Performance
- GPU acceleration: 1.92× speedup over CPU multi-threaded
- Throughput: 10.9 million points/second on NVIDIA RTX 2060
- Tested with 147.9M points, 62.8M grid cells
- Multi-threaded CPU execution with OpenMP
- Hybrid mode combining CPU routing and GPU accumulation

[0.1.0]: https://github.com/pcr-dev/pcr/releases/tag/v0.1.0
