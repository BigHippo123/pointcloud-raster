"""
PCR - Point Cloud Rasterization Library

High-performance GPU-accelerated point cloud rasterization to GeoTIFF.
"""

__version__ = "0.1.0"

# Read version from file if available
import os
_version_file = os.path.join(os.path.dirname(__file__), "..", "..", "VERSION")
if os.path.exists(_version_file):
    with open(_version_file) as f:
        __version__ = f.read().strip()

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
    GlyphType,

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

    # Pipeline / Glyph
    GlyphSpec,
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


# ---------------------------------------------------------------------------
# Convenience helpers for glyph ReductionSpec construction
# ---------------------------------------------------------------------------

def gaussian_splat_spec(
    value_channel,
    sigma_x_channel="",
    sigma_y_channel="",
    rotation_channel="",
    default_sigma=1.0,
    default_sigma_x=None,
    default_sigma_y=None,
    default_rotation=0.0,
    max_radius_cells=32.0,
    output_band_name=None,
):
    """
    Return a ReductionSpec configured for Gaussian glyph splatting.

    Each point paints a Gaussian footprint across nearby cells, producing smooth
    interpolated output even from sparse point clouds.

    Parameters
    ----------
    value_channel : str
        Channel name to reduce (e.g., "intensity", "z").
    sigma_x_channel : str, optional
        Per-point channel for sigma along X (world units). Empty = use default_sigma_x.
    sigma_y_channel : str, optional
        Per-point channel for sigma along Y (world units). Empty = use default_sigma_y.
    rotation_channel : str, optional
        Per-point channel for ellipse rotation (radians). Empty = 0.
    default_sigma : float
        Default sigma in world units when no per-point channel is set.
    default_sigma_x : float, optional
        Override default sigma along X (default: default_sigma).
    default_sigma_y : float, optional
        Override default sigma along Y (default: default_sigma).
    default_rotation : float
        Default rotation angle in radians.
    max_radius_cells : float
        Footprint is clamped to this many cells in each direction.
    output_band_name : str, optional
        Name for the output raster band.

    Returns
    -------
    ReductionSpec
    """
    spec = ReductionSpec()
    spec.value_channel = value_channel
    spec.type = ReductionType.WeightedAverage
    spec.glyph.type = GlyphType.Gaussian
    spec.glyph.sigma_x_channel = sigma_x_channel
    spec.glyph.sigma_y_channel = sigma_y_channel
    spec.glyph.rotation_channel = rotation_channel
    spec.glyph.default_sigma_x = default_sigma_x if default_sigma_x is not None else default_sigma
    spec.glyph.default_sigma_y = default_sigma_y if default_sigma_y is not None else default_sigma
    spec.glyph.default_rotation = default_rotation
    spec.glyph.max_radius_cells = max_radius_cells
    if output_band_name:
        spec.output_band_name = output_band_name
    return spec


def line_splat_spec(
    value_channel,
    direction_channel="",
    half_length_channel="",
    default_direction=0.0,
    default_half_length=1.0,
    max_radius_cells=32.0,
    output_band_name=None,
):
    """
    Return a ReductionSpec configured for Line glyph splatting.

    Each point paints a 1-pixel-wide Bresenham line segment centered on the point,
    oriented along `direction` with total length 2 * `half_length`.

    Parameters
    ----------
    value_channel : str
        Channel name to reduce.
    direction_channel : str, optional
        Per-point channel for line direction (radians, 0=East). Empty = default_direction.
    half_length_channel : str, optional
        Per-point channel for half-length (world units). Empty = default_half_length.
    default_direction : float
        Default direction in radians when no per-point channel is set.
    default_half_length : float
        Default half-length in world units.
    max_radius_cells : float
        Footprint is clamped to this many cells in each direction.
    output_band_name : str, optional
        Name for the output raster band.

    Returns
    -------
    ReductionSpec
    """
    spec = ReductionSpec()
    spec.value_channel = value_channel
    spec.type = ReductionType.WeightedAverage
    spec.glyph.type = GlyphType.Line
    spec.glyph.direction_channel = direction_channel
    spec.glyph.half_length_channel = half_length_channel
    spec.glyph.default_direction = default_direction
    spec.glyph.default_half_length = default_half_length
    spec.glyph.max_radius_cells = max_radius_cells
    if output_band_name:
        spec.output_band_name = output_band_name
    return spec


__all__ = [
    # Enums
    'DataType',
    'ReductionType',
    'MemoryLocation',
    'ExecutionMode',
    'StatusCode',
    'CompareOp',
    'PointCloudFormat',
    'GlyphType',

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

    # Pipeline / Glyph
    'GlyphSpec',
    'ReductionSpec',
    'PipelineConfig',
    'ProgressInfo',
    'Pipeline',

    # Convenience glyph helpers
    'gaussian_splat_spec',
    'line_splat_spec',

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
