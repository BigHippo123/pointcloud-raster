"""
PCR - Point Cloud Rasterization Library

High-performance GPU-accelerated point cloud rasterization to GeoTIFF.
"""

__version__ = "0.1.0"

# Import C++ extension module
from ._pcr import (
    # Enums
    DataType,
    ReductionType,
    MemoryLocation,
    ExecutionMode,
    StatusCode,
    CompareOp,
    PointCloudFormat,

    # Core types
    BBox,
    CRS,
    NoDataPolicy,
    TileIndex,
    Status,
    ChannelDesc,
    BandDesc,

    # Grid and GridConfig
    GridConfig,
    Grid,

    # PointCloud
    PointCloud,

    # Filter
    FilterPredicate,
    FilterSpec,

    # Pipeline
    ReductionSpec,
    PipelineConfig,
    ProgressInfo,
    Pipeline,

    # I/O - GeoTIFF
    GeoTiffOptions,
    write_geotiff,
    read_geotiff_info,

    # I/O - Point Cloud
    PointCloudInfo,
    read_point_cloud,
    write_point_cloud,
    read_point_cloud_info,
    PointCloudReader,
)

__all__ = [
    # Enums
    'DataType',
    'ReductionType',
    'MemoryLocation',
    'ExecutionMode',
    'StatusCode',
    'CompareOp',
    'PointCloudFormat',

    # Core types
    'BBox',
    'CRS',
    'NoDataPolicy',
    'TileIndex',
    'Status',
    'ChannelDesc',
    'BandDesc',

    # Grid and GridConfig
    'GridConfig',
    'Grid',

    # PointCloud
    'PointCloud',

    # Filter
    'FilterPredicate',
    'FilterSpec',

    # Pipeline
    'ReductionSpec',
    'PipelineConfig',
    'ProgressInfo',
    'Pipeline',

    # I/O - GeoTIFF
    'GeoTiffOptions',
    'write_geotiff',
    'read_geotiff_info',

    # I/O - Point Cloud
    'PointCloudInfo',
    'read_point_cloud',
    'write_point_cloud',
    'read_point_cloud_info',
    'PointCloudReader',
]
