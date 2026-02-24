#include "pcr/io/tile_state_io.h"
#include <fstream>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <sys/stat.h>

namespace pcr {

// ===========================================================================
// Constants
// ===========================================================================

static constexpr uint32_t MAGIC_PCRT = 0x54524350;  // "PCRT" in little-endian
static constexpr uint32_t FORMAT_VERSION = 1;

// Header size: magic(4) + version(4) + tile_row(4) + tile_col(4) +
//              cols(4) + rows(4) + state_floats(4) + reduction(1) + reserved(7)
static constexpr size_t HEADER_SIZE = 36;

// ===========================================================================
// Header structure
// ===========================================================================

#pragma pack(push, 1)
struct TileStateHeader {
    uint32_t magic;
    uint32_t version;
    int32_t  tile_row;
    int32_t  tile_col;
    int32_t  cols;
    int32_t  rows;
    int32_t  state_floats;
    uint8_t  reduction;
    uint8_t  reserved[7];
};
#pragma pack(pop)

static_assert(sizeof(TileStateHeader) == HEADER_SIZE, "Header size mismatch");

// ===========================================================================
// Implementation
// ===========================================================================

Status write_tile_state(
    const std::string& path,
    TileIndex tile,
    int cols, int rows,
    int state_floats,
    ReductionType type,
    const float* state)
{
    if (!state) {
        return Status::error(StatusCode::InvalidArgument, "null state pointer");
    }

    if (cols <= 0 || rows <= 0 || state_floats <= 0) {
        return Status::error(StatusCode::InvalidArgument, "invalid dimensions");
    }

    // Prepare header
    TileStateHeader header;
    header.magic        = MAGIC_PCRT;
    header.version      = FORMAT_VERSION;
    header.tile_row     = tile.row;
    header.tile_col     = tile.col;
    header.cols         = cols;
    header.rows         = rows;
    header.state_floats = state_floats;
    header.reduction    = static_cast<uint8_t>(type);
    std::memset(header.reserved, 0, sizeof(header.reserved));

    // Open file for writing
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return Status::error(StatusCode::IoError, "failed to open file for writing: " + path);
    }

    // Write header
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if (!ofs) {
        return Status::error(StatusCode::IoError, "failed to write header");
    }

    // Write state buffer
    const int64_t num_floats = static_cast<int64_t>(state_floats) * cols * rows;
    const size_t bytes = num_floats * sizeof(float);
    ofs.write(reinterpret_cast<const char*>(state), bytes);
    if (!ofs) {
        return Status::error(StatusCode::IoError, "failed to write state data");
    }

    ofs.close();
    return Status::success();
}

Status read_tile_state_header(
    const std::string& path,
    TileIndex& tile,
    int& cols, int& rows,
    int& state_floats,
    ReductionType& type)
{
    // Check if file exists
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return Status::error(StatusCode::IoError, "file not found: " + path);
    }

    // Open file
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to open file: " + path);
    }

    // Read header
    TileStateHeader header;
    ifs.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to read header");
    }

    // Validate magic
    if (header.magic != MAGIC_PCRT) {
        return Status::error(StatusCode::IoError, "invalid magic number (not a PCRT file)");
    }

    // Validate version
    if (header.version != FORMAT_VERSION) {
        std::ostringstream oss;
        oss << "unsupported version " << header.version << " (expected " << FORMAT_VERSION << ")";
        return Status::error(StatusCode::IoError, oss.str());
    }

    // Validate dimensions
    if (header.cols <= 0 || header.rows <= 0 || header.state_floats <= 0) {
        return Status::error(StatusCode::IoError, "invalid dimensions in header");
    }

    // Extract header fields
    tile.row      = header.tile_row;
    tile.col      = header.tile_col;
    cols          = header.cols;
    rows          = header.rows;
    state_floats  = header.state_floats;
    type          = static_cast<ReductionType>(header.reduction);

    return Status::success();
}

Status read_tile_state(
    const std::string& path,
    TileIndex& tile,
    int& cols, int& rows,
    int& state_floats,
    ReductionType& type,
    float* state)
{
    if (!state) {
        return Status::error(StatusCode::InvalidArgument, "null state pointer");
    }

    // Read header first
    Status s = read_tile_state_header(path, tile, cols, rows, state_floats, type);
    if (!s.ok()) {
        return s;
    }

    // Open file again to read body
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to open file: " + path);
    }

    // Skip header
    ifs.seekg(HEADER_SIZE, std::ios::beg);
    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to seek past header");
    }

    // Read state buffer
    const int64_t num_floats = static_cast<int64_t>(state_floats) * cols * rows;
    const size_t bytes = num_floats * sizeof(float);
    ifs.read(reinterpret_cast<char*>(state), bytes);
    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to read state data");
    }

    // Verify we read all expected data
    if (ifs.gcount() != static_cast<std::streamsize>(bytes)) {
        return Status::error(StatusCode::IoError, "incomplete state data (file truncated?)");
    }

    return Status::success();
}

std::string tile_state_filename(const std::string& dir, TileIndex tile) {
    std::ostringstream oss;
    if (!dir.empty()) {
        oss << dir;
        // Add trailing slash if not present
        if (dir.back() != '/') {
            oss << '/';
        }
    }
    oss << "tile_"
        << std::setw(4) << std::setfill('0') << tile.row << "_"
        << std::setw(4) << std::setfill('0') << tile.col
        << ".pcrt";
    return oss.str();
}

} // namespace pcr
