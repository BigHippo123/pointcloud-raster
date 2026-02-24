# PCR Python API Documentation

## Overview

The PCR Python API provides comprehensive bindings to the C++ library using pybind11, with seamless NumPy integration for efficient data manipulation.

## Installation

```bash
# Build with Python bindings
cd build
cmake .. -DPCR_BUILD_PYTHON=ON -DPCR_ENABLE_CUDA=OFF
make -j$(nproc)

# The Python module will be built at: python/pcr/_pcr.so
```

## Quick Start

```python
import sys
sys.path.insert(0, 'path/to/python')
import pcr
import numpy as np

# Create a point cloud
cloud = pcr.PointCloud.create(1000)
cloud.add_channel("intensity", pcr.DataType.Float32)

# Set coordinates
x = np.random.uniform(0, 100, 500)
y = np.random.uniform(0, 100, 500)
cloud.set_x_array(x)
cloud.set_y_array(y)

# Set intensity values
intensity = np.random.random(500).astype(np.float32)
cloud.set_channel_array_f32("intensity", intensity)
cloud.set_crs(pcr.CRS.from_epsg(32633))

# Configure pipeline
config = pcr.PipelineConfig()
config.grid.bounds.min_x = 0.0
config.grid.bounds.min_y = 0.0
config.grid.bounds.max_x = 100.0
config.grid.bounds.max_y = 100.0
config.grid.cell_size_x = 1.0
config.grid.cell_size_y = -1.0
config.grid.crs = pcr.CRS.from_epsg(32633)
config.grid.compute_dimensions()

# Add reduction
reduction = pcr.ReductionSpec()
reduction.value_channel = "intensity"
reduction.type = pcr.ReductionType.Average
reduction.output_band_name = "mean_intensity"
config.reductions.append(reduction)

# Set output
config.output_path = "/tmp/output.tif"
config.state_dir = "/tmp/pcr_tiles"

# Run pipeline
pipeline = pcr.Pipeline.create(config)
pipeline.validate()
pipeline.ingest(cloud)
pipeline.finalize()

# Access result
result = pipeline.result()
intensity_arr = result.band_array(0)  # NumPy array
print(f"Mean: {np.nanmean(intensity_arr):.2f}")
```

## API Reference

### Core Types

#### BBox
```python
bbox = pcr.BBox()
bbox.min_x = 0.0
bbox.min_y = 0.0
bbox.max_x = 100.0
bbox.max_y = 100.0
width = bbox.width()
height = bbox.height()
```

#### CRS (Coordinate Reference System)
```python
crs = pcr.CRS.from_epsg(4326)  # WGS84
crs = pcr.CRS.from_wkt(wkt_string)
is_proj = crs.is_projected()
```

#### GridConfig
```python
config = pcr.GridConfig()
config.bounds = bbox
config.cell_size_x = 1.0
config.cell_size_y = -1.0
config.crs = crs
config.compute_dimensions()  # Calculate width, height, tiles
```

### PointCloud

#### Creation
```python
cloud = pcr.PointCloud.create(capacity=10000)
```

#### Adding Channels
```python
cloud.add_channel("intensity", pcr.DataType.Float32)
cloud.add_channel("class", pcr.DataType.Float32)
cloud.add_channel("return_num", pcr.DataType.Int32)
```

#### Data Access (NumPy)
```python
# Get arrays (zero-copy, references C++ memory)
x = cloud.x_array()
y = cloud.y_array()
intensity = cloud.channel_array_f32("intensity")

# Set arrays (copies data to C++)
cloud.set_x_array(x_np_array)
cloud.set_y_array(y_np_array)
cloud.set_channel_array_f32("intensity", intensity_np_array)
```

#### Properties
```python
count = cloud.count()
capacity = cloud.capacity()
crs = cloud.crs()
channels = cloud.channel_names()
has_channel = cloud.has_channel("intensity")
```

### Grid

#### Creation
```python
band = pcr.BandDesc()
band.name = "elevation"
band.dtype = pcr.DataType.Float32

grid = pcr.Grid.create(cols=100, rows=100, bands=[band])
```

#### Data Access (NumPy)
```python
# Get band as NumPy array (zero-copy)
arr = grid.band_array(band_index)

# Modify data
arr[:] = np.random.random(arr.shape)

# Or set entire band
grid.set_band_array(band_index, new_array)
```

#### Operations
```python
grid.fill(0.0)  # Fill all bands
grid.fill_band(0, 42.0)  # Fill specific band
```

### Pipeline

#### Configuration
```python
config = pcr.PipelineConfig()

# Grid setup
config.grid = grid_config

# Reductions
reduction = pcr.ReductionSpec()
reduction.value_channel = "intensity"
reduction.type = pcr.ReductionType.Average
reduction.output_band_name = "mean_intensity"
config.reductions.append(reduction)

# Optional filter
filter_spec = pcr.FilterSpec()
filter_spec.add("class", pcr.CompareOp.Equal, 2.0)  # Ground only
config.filter = filter_spec

# Output
config.output_path = "/tmp/output.tif"
config.state_dir = "/tmp/tiles"
config.write_cog = True  # Cloud-Optimized GeoTIFF
```

#### Execution
```python
pipeline = pcr.Pipeline.create(config)
pipeline.validate()

# Process clouds
for cloud in clouds:
    pipeline.ingest(cloud)

# Finalize
pipeline.finalize()

# Get result
result = pipeline.result()
stats = pipeline.stats()
```

#### Progress Callbacks
```python
def progress_callback(info):
    print(f"Processed {info.points_processed} points in {info.elapsed_seconds:.1f}s")
    return True  # Return False to cancel

pipeline.set_progress_callback(progress_callback)
```

### Filtering

```python
# Create filter
filt = pcr.FilterSpec()
filt.add("class", pcr.CompareOp.Equal, 2.0)
filt.add("intensity", pcr.CompareOp.Greater, 50.0)
filt.add_in_set("return_num", [1.0, 2.0])  # Only first two returns
```

### I/O Functions

#### Point Cloud I/O
```python
# Read
cloud = pcr.read_point_cloud("input.pcrp")
info = pcr.read_point_cloud_info("input.pcrp")

# Write
pcr.write_point_cloud("output.pcrp", cloud)

# Streaming reader
reader = pcr.PointCloudReader.open("large_file.pcrp")
info = reader.info()
while not reader.eof():
    count = reader.read_chunk(cloud_buffer, max_points=10000)
```

#### GeoTIFF I/O
```python
# Write
options = pcr.GeoTiffOptions()
options.compress = "DEFLATE"
options.cloud_optimized = True
pcr.write_geotiff("output.tif", grid, grid_config, options)

# Read info
width, height, nbands, crs, bounds = pcr.read_geotiff_info("input.tif")
```

## Enums

### ReductionType
- `Sum` - Sum of values
- `Max` - Maximum value
- `Min` - Minimum value
- `Average` - Mean value
- `Count` - Point count
- `WeightedAverage` - Weighted mean (requires weight channel)
- `MostRecent` - Most recent value (requires timestamp)
- `Median` - Median value
- `Percentile` - Percentile value

### DataType
- `Float32`, `Float64`
- `Int32`, `UInt32`
- `Int16`, `UInt16`
- `UInt8`

### CompareOp
- `Equal`, `NotEqual`
- `Less`, `LessEqual`
- `Greater`, `GreaterEqual`
- `InSet`, `NotInSet`

### MemoryLocation
- `Host` - CPU memory
- `HostPinned` - Pinned CPU memory (for GPU transfer)
- `Device` - GPU memory

## Examples

See `examples/python/basic_rasterize.py` for a complete working example.

## NumPy Integration

The Python bindings provide zero-copy access to C++ memory through NumPy arrays:

```python
# Grid bands
band_array = grid.band_array(0)  # Returns np.ndarray pointing to C++ memory
band_array[:, :] = new_values  # Modify in-place

# PointCloud
x = cloud.x_array()  # Zero-copy access to coordinates
intensity = cloud.channel_array_f32("intensity")  # Zero-copy channel access
```

**Important:** Arrays returned by these methods reference C++ memory. The C++ objects
(Grid, PointCloud) must remain alive while NumPy arrays are in use. This is handled
automatically by pybind11 keep_alive policies.

## Error Handling

C++ Status errors are automatically converted to Python exceptions:

```python
try:
    pipeline.validate()
    pipeline.ingest(cloud)
    pipeline.finalize()
except RuntimeError as e:
    print(f"Pipeline error: {e}")
```

## Performance Tips

1. **Use NumPy for bulk operations**: Setting coordinates/channels with NumPy arrays
   is much faster than setting individual points.

2. **Pre-allocate capacity**: Create PointCloud with sufficient capacity to avoid
   reallocations.

3. **Batch processing**: Process multiple point clouds in a single pipeline run
   for better tile cache utilization.

4. **Memory budgets**: Set appropriate `host_cache_budget` in PipelineConfig to
   control memory usage.

5. **Cloud-Optimized GeoTIFF**: Enable `write_cog=True` for better performance
   when reading outputs later.

## Thread Safety

The Python bindings are **not thread-safe**. Create separate Pipeline instances
for parallel processing. PointCloud and Grid objects should not be accessed from
multiple threads simultaneously.

## Known Limitations

1. **CPU-only**: Current bindings support CPU-only builds (PCR_ENABLE_CUDA=OFF).
   GPU support will be added in a future release.

2. **WeightedAverage and MostRecent**: These reduction types are defined but not
   yet fully implemented in the C++ backend.

3. **Point filtering**: Filtering with PointCloud views is not yet supported.

4. **CRS reprojection**: Auto-reprojection (`auto_reproject` flag) is not yet
   implemented.
