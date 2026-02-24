#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include "pcr/engine/glyph.h"
#include "pcr/engine/tile_router.h"

namespace pcr {

// ---------------------------------------------------------------------------
// accumulate_glyph — paint a weighted glyph footprint for each point
//
// Each point in `batch` splatts its value across all cells within its glyph
// region, weighting each cell by the glyph kernel weight.
//
// Supported glyph types:
//   Point    — identical to regular accumulate (weight=1.0 for the single cell)
//   Line     — Bresenham line segment; each cell on the line gets weight 1.0
//   Gaussian — Gaussian kernel; weight = exp(-0.5 * ((dx/sx)²+(dy/sy)²))
//
// Supported reduction types (others return NotImplemented):
//   WeightedAverage — state[0*N+cell] += val*w; state[1*N+cell] += w
//   Average         — same as WeightedAverage (weighted average semantics)
//   Sum             — state[cell] += val*w
//   Count           — state[cell] += w
//
// State layout must already be initialized (zeroed / identity) by TileManager.
//
// batch.coord_x / coord_y must be non-null for Line and Gaussian glyphs.
// ---------------------------------------------------------------------------
Status accumulate_glyph(
    const GlyphSpec&  spec,
    ReductionType     reduction_type,
    const TileBatch&  batch,
    float*            state,
    int64_t           tile_cells,
    const GridConfig& grid_cfg,
    int               tile_col_origin,
    int               tile_row_origin,
    int               tile_width_actual,
    int               tile_height_actual,
    void*             stream = nullptr);

} // namespace pcr
