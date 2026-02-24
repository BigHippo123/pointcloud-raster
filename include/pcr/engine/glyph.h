#pragma once

#include <string>
#include <cstdint>

namespace pcr {

// ---------------------------------------------------------------------------
// GlyphType — how a point's value is splatted onto the raster
// ---------------------------------------------------------------------------
enum class GlyphType : uint8_t {
    Point,     // existing: 1-cell scatter (no overhead, default)
    Line,      // direction + half_length → Bresenham line segment
    Gaussian,  // sigma_x, sigma_y, rotation → Gaussian kernel footprint
};

// ---------------------------------------------------------------------------
// GlyphSpec — per-reduction glyph configuration
// ---------------------------------------------------------------------------
struct GlyphSpec {
    GlyphType   type = GlyphType::Point;

    // --- Line params ---
    // Channel names reference Float32 channels in the PointCloud.
    // If channel name is empty or the channel is absent, the default_ value is used.
    std::string direction_channel;    float default_direction   = 0.0f; // radians (0 = East/+X, π/2 = North/+Y)
    std::string half_length_channel;  float default_half_length = 1.0f; // world units

    // --- Gaussian params ---
    std::string sigma_x_channel;     float default_sigma_x = 1.0f;    // world units
    std::string sigma_y_channel;     float default_sigma_y = 1.0f;    // world units
    std::string rotation_channel;    float default_rotation = 0.0f;   // radians

    // --- Safety cap ---
    // Clamp footprint to this many cells in each direction.
    // Prevents runaway work for large sigma / half_length values.
    float max_radius_cells = 32.0f;

    // If true, each point's glyph contribution is normalized so its total
    // weight across all cells sums to 1.0.
    bool normalize_weights = false;
};

} // namespace pcr
