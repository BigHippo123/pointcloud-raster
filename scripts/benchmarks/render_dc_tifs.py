#!/usr/bin/env python3
"""
Render DC LiDAR TIF outputs into a single comparison PNG for the benchmark report.

Reads dc_lidar_*_*.tif files produced by test_dc_lidar.py and generates a
side-by-side grid showing how each glyph type renders the same dataset.

Usage:
    python3 render_dc_tifs.py --tif-dir /path/to/tifs --output comparison.png [--mode cpu-mt]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Try matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
except ImportError:
    print("matplotlib not available — skipping DC comparison render", file=sys.stderr)
    sys.exit(0)

# Try pcr for TIF reading, fall back to rasterio
def read_tif(path: Path):
    try:
        sys.path.insert(0, str(path.parent.parent.parent / "python"))
        import pcr
        g = pcr.Grid.from_file(str(path))
        arr = g.band_array(0).copy()
        return arr
    except Exception:
        pass
    try:
        import rasterio
        with rasterio.open(str(path)) as src:
            arr = src.read(1).astype(np.float32)
            arr[arr == src.nodata] = np.nan if src.nodata else arr
            return arr
    except Exception:
        return None


# Ordered glyph display layout
_GLYPH_ORDER = [
    ("point",       "Point"),
    ("line-2",      "Line  hl=2"),
    ("line-5",      "Line  hl=5"),
    ("gaussian-1",  "Gaussian  σ=1"),
    ("gaussian-3",  "Gaussian  σ=3"),
    ("gaussian-5",  "Gaussian  σ=5"),
    ("gaussian-10", "Gaussian  σ=10"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tif-dir", required=True,
                        help="Directory containing dc_lidar_*.tif files")
    parser.add_argument("--output", required=True,
                        help="Output PNG path")
    parser.add_argument("--mode", default="cpu-mt",
                        help="Which mode's TIFs to render (default: cpu-mt)")
    args = parser.parse_args()

    tif_dir = Path(args.tif_dir)
    output  = Path(args.output)

    # Collect available TIFs for the requested mode
    panels = []
    for glyph_key, label in _GLYPH_ORDER:
        tif_path = tif_dir / f"dc_lidar_{args.mode}_{glyph_key}.tif"
        if tif_path.exists():
            arr = read_tif(tif_path)
            if arr is not None:
                panels.append((label, arr))

    if not panels:
        print(f"No TIF files found in {tif_dir} for mode={args.mode}", file=sys.stderr)
        sys.exit(1)

    n = len(panels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig_w = ncols * 5
    fig_h = nrows * 4.5

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    fig.suptitle(
        f"DC LiDAR — Glyph Comparison  (mode: {args.mode})",
        fontsize=13, color="white", fontweight="bold", y=1.01,
    )

    # Compute shared color range from the point glyph (first panel)
    ref_arr = panels[0][1]
    valid = ref_arr[~np.isnan(ref_arr)]
    vmin = float(np.percentile(valid, 2))  if len(valid) else 0
    vmax = float(np.percentile(valid, 98)) if len(valid) else 1
    cmap = plt.colormaps["terrain"]
    cmap.set_bad("#0d0d1a")

    for idx, (label, arr) in enumerate(panels):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        # Downsample for display if very large
        h, w = arr.shape
        factor = max(1, max(h, w) // 1200)
        if factor > 1:
            arr = arr[::factor, ::factor]
        valid_cells = np.sum(~np.isnan(arr))
        coverage = 100 * valid_cells / arr.size
        ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax,
                  interpolation="nearest", origin="upper", aspect="equal")
        ax.set_title(f"{label}\n{coverage:.0f}% covered",
                     fontsize=8, color="white", pad=3)
        ax.axis("off")

    fig.tight_layout(pad=0.8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=110, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  DC comparison saved: {output}")


if __name__ == "__main__":
    main()
