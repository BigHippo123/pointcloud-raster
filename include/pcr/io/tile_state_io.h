#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include <string>

namespace pcr {

// ---------------------------------------------------------------------------
// Tile state file format (binary):
//
//   Header (fixed size):
//     magic:        uint32  "PCRT"
//     version:      uint32  1
//     tile_row:     int32
//     tile_col:     int32
//     cols:         int32   tile width in cells
//     rows:         int32   tile height in cells
//     state_floats: int32   number of float fields per cell
//     reduction:    uint8   ReductionType enum
//     reserved:     uint8[7]
//
//   Body:
//     float[state_floats * cols * rows]   band-sequential layout
//
// ---------------------------------------------------------------------------

/// Write tile state to disk.
/// `state` is a host pointer, size = state_floats * cols * rows floats.
Status write_tile_state(const std::string& path,
                        TileIndex tile,
                        int cols, int rows,
                        int state_floats,
                        ReductionType type,
                        const float* state);

/// Read tile state from disk.
/// `state` must be pre-allocated, size = state_floats * cols * rows floats.
/// Returns IoError if file doesn't exist or is corrupt.
Status read_tile_state(const std::string& path,
                       TileIndex& tile,
                       int& cols, int& rows,
                       int& state_floats,
                       ReductionType& type,
                       float* state);

/// Peek at tile state header without reading body.
Status read_tile_state_header(const std::string& path,
                              TileIndex& tile,
                              int& cols, int& rows,
                              int& state_floats,
                              ReductionType& type);

/// Construct the file path for a tile state file.
std::string tile_state_filename(const std::string& dir, TileIndex tile);

} // namespace pcr
