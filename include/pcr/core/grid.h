#pragma once

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include <string>
#include <vector>
#include <memory>

namespace pcr {

// ---------------------------------------------------------------------------
// BandDesc — describes one output band or internal state band
// ---------------------------------------------------------------------------
struct BandDesc {
    std::string name;
    DataType    dtype = DataType::Float32;
    bool        is_state = false;   // true = internal reduction state, not a final output band
};

// ---------------------------------------------------------------------------
// Grid — 2D multi-band raster buffer.
//
// Can represent:
//   1. Final output: one or more bands of reduced values.
//   2. Running reduction state: compound state per cell (e.g., sum+count for average).
//
// Data can live on Host, HostPinned, or Device.
// For tiled processing, each Grid instance represents ONE tile.
// ---------------------------------------------------------------------------
class Grid {
public:
    Grid() = default;
    ~Grid();

    /// Create a grid (single tile or full grid).
    /// `cols` x `rows` cells, with specified bands.
    static std::unique_ptr<Grid> create(int cols, int rows,
                                        const std::vector<BandDesc>& bands,
                                        MemoryLocation loc = MemoryLocation::Host);

    /// Create a grid matching a tile within a GridConfig.
    static std::unique_ptr<Grid> create_for_tile(const GridConfig& config,
                                                 TileIndex tile,
                                                 const std::vector<BandDesc>& bands,
                                                 MemoryLocation loc = MemoryLocation::Host);

    // -- Band access --------------------------------------------------------

    int         num_bands() const;
    BandDesc    band_desc(int band_index) const;
    int         band_index(const std::string& name) const;  // -1 if not found

    /// Raw pointer to band data (row-major, contiguous).
    void*       band_data(int band_index);
    const void* band_data(int band_index) const;

    /// Typed convenience
    float*       band_f32(int band_index);
    const float* band_f32(int band_index) const;
    float*       band_f32(const std::string& name);
    const float* band_f32(const std::string& name) const;

    // -- Properties ---------------------------------------------------------

    int            cols()     const;
    int            rows()     const;
    int64_t        cell_count() const;
    MemoryLocation location() const;

    // -- Initialize all bands to nodata / identity values -------------------

    /// Fill all cells in all bands with a value (typically nodata or op identity).
    Status fill(float value);

    /// Fill a specific band.
    Status fill_band(int band_index, float value);

    // -- Device transfer ----------------------------------------------------

    std::unique_ptr<Grid> to(MemoryLocation dst) const;
    std::unique_ptr<Grid> to_device_async(void* cuda_stream) const;

    /// Copy data from another grid into this grid (must have same dimensions/bands).
    /// Supports cross-location copy (device→host, etc.).
    Status copy_from(const Grid& other, void* cuda_stream = nullptr);

    // -- Nodata mask --------------------------------------------------------

    /// Generate a boolean mask: true where cell has data (not nodata).
    /// Examines band 0 by default.
    std::vector<uint8_t> valid_mask(int band_index = 0) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace pcr
