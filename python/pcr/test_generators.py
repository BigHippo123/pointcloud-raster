"""
Synthetic point cloud generators for testing and visualization.

Each generator creates a PointCloud with a known, visually recognizable pattern
that can be easily verified in the output raster.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import pcr


# Small epsilon to avoid points exactly on bbox boundaries (floating-point safety)
_BBOX_EPSILON = 1e-6


def _safe_uniform(low: float, high: float, size: int) -> np.ndarray:
    """
    Generate random uniform points with a small epsilon margin from boundaries.

    This prevents floating-point edge cases where points exactly at max boundary
    could round to be outside the grid when computing cell indices.
    """
    return np.random.uniform(low + _BBOX_EPSILON, high - _BBOX_EPSILON, size)


# ==============================================================================
# VISUAL PATTERN GENERATORS (easy to verify by eye)
# ==============================================================================

def generate_checkerboard(
    bbox: pcr.BBox,
    cell_size: float,
    points_per_cell: int = 10,
    square_size: int = 8,
    value_low: float = 0.0,
    value_high: float = 100.0,
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate a checkerboard pattern.

    Creates alternating squares of high and low density points that produce
    a clear checkerboard pattern in the output raster.

    Args:
        bbox: Bounding box for the point cloud
        cell_size: Output grid cell size (meters or CRS units)
        points_per_cell: Average number of points per output cell
        square_size: Number of cells per checkerboard square
        value_low: Value for "black" squares
        value_high: Value for "white" squares
        seed: Random seed for reproducibility

    Returns:
        (PointCloud, metadata dict with expected pattern info)
    """
    np.random.seed(seed)

    # Calculate grid dimensions
    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)

    # Generate points uniformly distributed in space
    num_points = width * height * points_per_cell
    points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
    points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)

    # Determine which checkerboard square each point falls into
    grid_col = ((points_x - bbox.min_x) / cell_size).astype(int)
    grid_row = ((points_y - bbox.min_y) / cell_size).astype(int)

    square_col = grid_col // square_size
    square_row = grid_row // square_size

    # Checkerboard pattern: (row + col) % 2 determines color
    is_white = (square_row + square_col) % 2 == 0

    values = np.where(is_white, value_high, value_low).astype(np.float32)

    # Create PointCloud
    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "checkerboard",
        "square_size": square_size,
        "value_low": value_low,
        "value_high": value_high,
        "num_points": num_points,
        "grid_size": (width, height),
    }

    return cloud, metadata


def generate_stripes(
    bbox: pcr.BBox,
    cell_size: float,
    points_per_cell: int = 10,
    stripe_width: int = 5,
    orientation: str = "horizontal",
    value_low: float = 0.0,
    value_high: float = 100.0,
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate stripe pattern.

    Args:
        bbox: Bounding box for the point cloud
        cell_size: Output grid cell size
        points_per_cell: Average points per cell
        stripe_width: Width of each stripe in cells
        orientation: "horizontal", "vertical", or "diagonal"
        value_low: Value for dark stripes
        value_high: Value for light stripes
        seed: Random seed

    Returns:
        (PointCloud, metadata)
    """
    np.random.seed(seed)

    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)
    num_points = width * height * points_per_cell

    points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
    points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)

    grid_col = ((points_x - bbox.min_x) / cell_size).astype(int)
    grid_row = ((points_y - bbox.min_y) / cell_size).astype(int)

    if orientation == "horizontal":
        stripe_idx = grid_row // stripe_width
    elif orientation == "vertical":
        stripe_idx = grid_col // stripe_width
    elif orientation == "diagonal":
        stripe_idx = (grid_row + grid_col) // stripe_width
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    is_light = stripe_idx % 2 == 0
    values = np.where(is_light, value_high, value_low).astype(np.float32)

    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "stripes",
        "orientation": orientation,
        "stripe_width": stripe_width,
        "value_low": value_low,
        "value_high": value_high,
        "num_points": num_points,
    }

    return cloud, metadata


def generate_bullseye(
    bbox: pcr.BBox,
    cell_size: float,
    points_per_cell: int = 10,
    num_rings: int = 5,
    value_low: float = 0.0,
    value_high: float = 100.0,
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate concentric circles (bullseye pattern).

    Args:
        bbox: Bounding box
        cell_size: Cell size
        points_per_cell: Points per cell
        num_rings: Number of concentric rings
        value_low: Dark ring value
        value_high: Light ring value
        seed: Random seed

    Returns:
        (PointCloud, metadata)
    """
    np.random.seed(seed)

    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)
    num_points = width * height * points_per_cell

    points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
    points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)

    # Center of bbox
    center_x = (bbox.min_x + bbox.max_x) / 2
    center_y = (bbox.min_y + bbox.max_y) / 2

    # Distance from center
    dx = points_x - center_x
    dy = points_y - center_y
    distance = np.sqrt(dx**2 + dy**2)

    # Maximum radius
    max_radius = min(bbox.width(), bbox.height()) / 2

    # Which ring does each point belong to?
    ring_width = max_radius / num_rings
    ring_idx = (distance / ring_width).astype(int)
    ring_idx = np.clip(ring_idx, 0, num_rings - 1)

    is_light = ring_idx % 2 == 0
    values = np.where(is_light, value_high, value_low).astype(np.float32)

    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "bullseye",
        "num_rings": num_rings,
        "value_low": value_low,
        "value_high": value_high,
        "num_points": num_points,
    }

    return cloud, metadata


def generate_gradient(
    bbox: pcr.BBox,
    cell_size: float,
    points_per_cell: int = 10,
    gradient_type: str = "linear",
    angle: float = 0.0,
    value_min: float = 0.0,
    value_max: float = 100.0,
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate smooth gradient pattern.

    Args:
        bbox: Bounding box
        cell_size: Cell size
        points_per_cell: Points per cell
        gradient_type: "linear" or "radial"
        angle: Angle in degrees for linear gradient (0 = left to right)
        value_min: Minimum value
        value_max: Maximum value
        seed: Random seed

    Returns:
        (PointCloud, metadata)
    """
    np.random.seed(seed)

    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)
    num_points = width * height * points_per_cell

    points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
    points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)

    if gradient_type == "linear":
        # Normalize coordinates to [0, 1]
        norm_x = (points_x - bbox.min_x) / bbox.width()
        norm_y = (points_y - bbox.min_y) / bbox.height()

        # Apply rotation
        angle_rad = np.radians(angle)
        gradient_coord = norm_x * np.cos(angle_rad) + norm_y * np.sin(angle_rad)
        gradient_coord = np.clip(gradient_coord, 0, 1)

    elif gradient_type == "radial":
        center_x = (bbox.min_x + bbox.max_x) / 2
        center_y = (bbox.min_y + bbox.max_y) / 2
        dx = points_x - center_x
        dy = points_y - center_y
        distance = np.sqrt(dx**2 + dy**2)
        max_radius = min(bbox.width(), bbox.height()) / 2
        gradient_coord = np.clip(distance / max_radius, 0, 1)
    else:
        raise ValueError(f"Unknown gradient type: {gradient_type}")

    values = (value_min + gradient_coord * (value_max - value_min)).astype(np.float32)

    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "gradient",
        "gradient_type": gradient_type,
        "angle": angle if gradient_type == "linear" else None,
        "value_min": value_min,
        "value_max": value_max,
        "num_points": num_points,
    }

    return cloud, metadata


def generate_text(
    bbox: pcr.BBox,
    cell_size: float,
    text: str = "PCR",
    points_per_cell: int = 10,
    value_background: float = 0.0,
    value_text: float = 100.0,
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate text pattern (simplified block letters).

    Creates points that spell out text in the output raster.
    Uses a simple block letter representation.

    Args:
        bbox: Bounding box
        cell_size: Cell size
        text: Text to display (limited character set)
        points_per_cell: Points per cell
        value_background: Background value
        value_text: Text value
        seed: Random seed

    Returns:
        (PointCloud, metadata)
    """
    np.random.seed(seed)

    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)
    num_points = width * height * points_per_cell

    points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
    points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)

    # Simple block letter patterns (5x5 grid, 1=filled, 0=empty)
    # Centered in output grid
    letter_patterns = {
        'P': np.array([
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
        ]),
        'C': np.array([
            [0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
        ]),
        'R': np.array([
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ]),
    }

    # Create a grid to mark which cells should have text
    grid = np.zeros((height, width), dtype=bool)

    # Calculate letter placement
    letter_height = 5
    letter_width = 5
    letter_spacing = 2
    total_width = len(text) * letter_width + (len(text) - 1) * letter_spacing

    # Center text
    start_col = max(0, (width - total_width) // 2)
    start_row = max(0, (height - letter_height) // 2)

    # Place each letter
    for i, char in enumerate(text.upper()):
        if char in letter_patterns:
            pattern = letter_patterns[char]
            col_offset = i * (letter_width + letter_spacing)

            for r in range(letter_height):
                for c in range(letter_width):
                    row_idx = start_row + r
                    col_idx = start_col + col_offset + c

                    if 0 <= row_idx < height and 0 <= col_idx < width:
                        grid[row_idx, col_idx] = pattern[r, c]

    # Assign values based on which cell each point falls into
    grid_col = ((points_x - bbox.min_x) / cell_size).astype(int)
    grid_row = ((points_y - bbox.min_y) / cell_size).astype(int)
    grid_col = np.clip(grid_col, 0, width - 1)
    grid_row = np.clip(grid_row, 0, height - 1)

    is_text = grid[grid_row, grid_col]
    values = np.where(is_text, value_text, value_background).astype(np.float32)

    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "text",
        "text": text,
        "value_background": value_background,
        "value_text": value_text,
        "num_points": num_points,
    }

    return cloud, metadata


def generate_shapes(
    bbox: pcr.BBox,
    cell_size: float,
    shape: str = "circle",
    points_per_cell: int = 10,
    value_inside: float = 100.0,
    value_outside: float = 0.0,
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate geometric shapes.

    Args:
        bbox: Bounding box
        cell_size: Cell size
        shape: "circle", "square", or "triangle"
        points_per_cell: Points per cell
        value_inside: Value inside shape
        value_outside: Value outside shape
        seed: Random seed

    Returns:
        (PointCloud, metadata)
    """
    np.random.seed(seed)

    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)
    num_points = width * height * points_per_cell

    points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
    points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)

    center_x = (bbox.min_x + bbox.max_x) / 2
    center_y = (bbox.min_y + bbox.max_y) / 2

    # Normalize coordinates to [-1, 1]
    norm_x = 2 * (points_x - center_x) / bbox.width()
    norm_y = 2 * (points_y - center_y) / bbox.height()

    if shape == "circle":
        radius = 0.7
        distance = np.sqrt(norm_x**2 + norm_y**2)
        inside = distance <= radius

    elif shape == "square":
        size = 0.7
        inside = (np.abs(norm_x) <= size) & (np.abs(norm_y) <= size)

    elif shape == "triangle":
        # Equilateral triangle pointing up
        inside = (norm_y <= 0.5) & (norm_y >= -0.8 * np.abs(norm_x) + 0.5)

    else:
        raise ValueError(f"Unknown shape: {shape}")

    values = np.where(inside, value_inside, value_outside).astype(np.float32)

    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "shape",
        "shape": shape,
        "value_inside": value_inside,
        "value_outside": value_outside,
        "num_points": num_points,
    }

    return cloud, metadata


# ==============================================================================
# TECHNICAL TEST PATTERNS (for correctness validation)
# ==============================================================================

def generate_uniform_grid(
    bbox: pcr.BBox,
    cell_size: float,
    points_per_cell: int = 10,
    value: float = 50.0,
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate uniformly distributed points with constant value.

    Perfect for testing that all reduction ops produce the expected value.

    Args:
        bbox: Bounding box
        cell_size: Cell size
        points_per_cell: Exact points per cell
        value: Constant value for all points
        seed: Random seed

    Returns:
        (PointCloud, metadata with expected output)
    """
    np.random.seed(seed)

    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)
    num_points = width * height * points_per_cell

    points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
    points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)
    values = np.full(num_points, value, dtype=np.float32)

    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "uniform",
        "expected_value": value,
        "expected_count": points_per_cell,
        "expected_sum": value * points_per_cell,
        "expected_min": value,
        "expected_max": value,
        "expected_average": value,
        "num_points": num_points,
    }

    return cloud, metadata


def generate_gaussian_clusters(
    bbox: pcr.BBox,
    cell_size: float,
    num_clusters: int = 5,
    points_per_cluster: int = 1000,
    cluster_std: float = 10.0,
    value_range: Tuple[float, float] = (0.0, 100.0),
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate Gaussian clusters at random locations.

    Tests merge behavior at varying densities.

    Args:
        bbox: Bounding box
        cell_size: Cell size
        num_clusters: Number of cluster centers
        points_per_cluster: Points per cluster
        cluster_std: Standard deviation of each cluster (in CRS units)
        value_range: (min, max) value range across clusters
        seed: Random seed

    Returns:
        (PointCloud, metadata)
    """
    np.random.seed(seed)

    all_x = []
    all_y = []
    all_values = []

    # Random cluster centers
    centers_x = _safe_uniform(bbox.min_x, bbox.max_x, num_clusters)
    centers_y = _safe_uniform(bbox.min_y, bbox.max_y, num_clusters)
    cluster_values = np.linspace(value_range[0], value_range[1], num_clusters)

    for i in range(num_clusters):
        # Generate cluster points (Gaussian distribution)
        x = np.random.normal(centers_x[i], cluster_std, points_per_cluster)
        y = np.random.normal(centers_y[i], cluster_std, points_per_cluster)

        # Clip to bbox
        x = np.clip(x, bbox.min_x, bbox.max_x)
        y = np.clip(y, bbox.min_y, bbox.max_y)

        values = np.full(points_per_cluster, cluster_values[i], dtype=np.float32)

        all_x.append(x)
        all_y.append(y)
        all_values.append(values)

    points_x = np.concatenate(all_x)
    points_y = np.concatenate(all_y)
    values = np.concatenate(all_values)

    num_points_total = len(points_x)
    cloud = pcr.PointCloud.create(num_points_total, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "gaussian_clusters",
        "num_clusters": num_clusters,
        "points_per_cluster": points_per_cluster,
        "cluster_std": cluster_std,
        "value_range": value_range,
        "num_points": num_points_total,
    }

    return cloud, metadata


def generate_planar_surface(
    bbox: pcr.BBox,
    cell_size: float,
    points_per_cell: int = 10,
    slope_x: float = 1.0,
    slope_y: float = 0.5,
    noise_std: float = 5.0,
    base_value: float = 50.0,
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate planar surface with optional noise.

    Value = base + slope_x * x + slope_y * y + noise
    Validates interpolation quality.

    Args:
        bbox: Bounding box
        cell_size: Cell size
        points_per_cell: Points per cell
        slope_x: Slope in x direction
        slope_y: Slope in y direction
        noise_std: Standard deviation of Gaussian noise
        base_value: Base value at origin
        seed: Random seed

    Returns:
        (PointCloud, metadata)
    """
    np.random.seed(seed)

    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)
    num_points = width * height * points_per_cell

    points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
    points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)

    # Planar surface
    values = base_value + slope_x * points_x + slope_y * points_y

    # Add noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, num_points)
        values += noise

    values = values.astype(np.float32)

    cloud = pcr.PointCloud.create(num_points, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "planar_surface",
        "slope_x": slope_x,
        "slope_y": slope_y,
        "noise_std": noise_std,
        "base_value": base_value,
        "num_points": num_points,
    }

    return cloud, metadata


def generate_edge_cases(
    bbox: pcr.BBox,
    cell_size: float,
    case: str = "single_point",
    seed: int = 42,
) -> Tuple[pcr.PointCloud, Dict[str, Any]]:
    """
    Generate edge case point clouds.

    Args:
        bbox: Bounding box
        cell_size: Cell size
        case: Edge case type:
            - "single_point": One point total
            - "cell_boundaries": Points exactly on cell edges
            - "sparse": 0-1 points per cell randomly
            - "dense": 10000+ points in single cell
            - "mixed_density": Mix of empty, sparse, and dense cells
        seed: Random seed

    Returns:
        (PointCloud, metadata)
    """
    np.random.seed(seed)

    width = int((bbox.max_x - bbox.min_x) / cell_size)
    height = int((bbox.max_y - bbox.min_y) / cell_size)

    if case == "single_point":
        points_x = np.array([(bbox.min_x + bbox.max_x) / 2])
        points_y = np.array([(bbox.min_y + bbox.max_y) / 2])
        values = np.array([50.0], dtype=np.float32)

    elif case == "cell_boundaries":
        # Points exactly on cell boundaries
        num_cells = min(width, height, 100)
        points_x = []
        points_y = []

        for i in range(num_cells):
            x = bbox.min_x + i * cell_size
            y = bbox.min_y + i * cell_size
            points_x.append(x)
            points_y.append(y)

        points_x = np.array(points_x)
        points_y = np.array(points_y)
        values = np.full(len(points_x), 50.0, dtype=np.float32)

    elif case == "sparse":
        # Randomly 0 or 1 point per cell
        num_cells = width * height
        num_points = int(num_cells * 0.3)  # 30% of cells get a point

        points_x = _safe_uniform(bbox.min_x, bbox.max_x, num_points)
        points_y = _safe_uniform(bbox.min_y, bbox.max_y, num_points)
        values = np.full(num_points, 50.0, dtype=np.float32)

    elif case == "dense":
        # 10000 points in center cell
        center_x = (bbox.min_x + bbox.max_x) / 2
        center_y = (bbox.min_y + bbox.max_y) / 2

        points_x = np.random.uniform(center_x, center_x + cell_size, 10000)
        points_y = np.random.uniform(center_y, center_y + cell_size, 10000)
        values = np.full(10000, 50.0, dtype=np.float32)

    elif case == "mixed_density":
        # Mix: some empty, some sparse, some very dense
        points_x = []
        points_y = []
        values_list = []

        for row in range(min(height, 20)):
            for col in range(min(width, 20)):
                cell_x = bbox.min_x + col * cell_size
                cell_y = bbox.min_y + row * cell_size

                # Density pattern
                if (row + col) % 3 == 0:
                    # Empty cell
                    continue
                elif (row + col) % 3 == 1:
                    # Sparse: 1-5 points
                    n = np.random.randint(1, 6)
                else:
                    # Dense: 100-1000 points
                    n = np.random.randint(100, 1001)

                x = np.random.uniform(cell_x, cell_x + cell_size, n)
                y = np.random.uniform(cell_y, cell_y + cell_size, n)
                v = np.full(n, 50.0, dtype=np.float32)

                points_x.append(x)
                points_y.append(y)
                values_list.append(v)

        points_x = np.concatenate(points_x) if points_x else np.array([])
        points_y = np.concatenate(points_y) if points_y else np.array([])
        values = np.concatenate(values_list) if values_list else np.array([])

    else:
        raise ValueError(f"Unknown edge case: {case}")

    num_points_total = len(points_x)
    cloud = pcr.PointCloud.create(num_points_total, pcr.MemoryLocation.Host)
    cloud.set_x_array(points_x)
    cloud.set_y_array(points_y)
    cloud.add_channel("value", pcr.DataType.Float32)
    cloud.set_channel_array_f32("value", values)

    metadata = {
        "pattern": "edge_case",
        "case": case,
        "num_points": len(points_x),
    }

    return cloud, metadata
