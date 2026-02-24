#include "pcr/core/grid.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace pcr {

// ---------------------------------------------------------------------------
// Grid::Impl — PIMPL implementation
// ---------------------------------------------------------------------------
struct Grid::Impl {
    int cols_ = 0;
    int rows_ = 0;
    std::vector<BandDesc> bands_;
    MemoryLocation location_ = MemoryLocation::Host;

    // Band-major storage: each band is a separate allocation
    // This simplifies band-wise operations and matches the reduction state layout
    std::vector<void*> band_buffers_;

    // Total cells per band
    int64_t cell_count() const {
        return static_cast<int64_t>(cols_) * rows_;
    }

    // Allocate memory for all bands
    Status allocate() {
        const int64_t cells = cell_count();

        for (const auto& band : bands_) {
            const size_t bytes = cells * data_type_size(band.dtype);
            void* ptr = nullptr;

            switch (location_) {
                case MemoryLocation::Host:
                    ptr = std::malloc(bytes);
                    if (!ptr) {
                        return Status::error(StatusCode::OutOfMemory,
                                           "Failed to allocate host memory for band");
                    }
                    break;

                case MemoryLocation::HostPinned:
                case MemoryLocation::Device:
#ifdef PCR_HAS_CUDA
                    // TODO: CUDA allocation when CUDA enabled
                    return Status::error(StatusCode::NotImplemented,
                                       "CUDA memory allocation not yet implemented");
#else
                    return Status::error(StatusCode::NotImplemented,
                                       "HostPinned and Device memory require CUDA build");
#endif
            }

            band_buffers_.push_back(ptr);
        }

        return Status::success();
    }

    // Free all memory
    void deallocate() {
        for (void* ptr : band_buffers_) {
            if (ptr) {
                switch (location_) {
                    case MemoryLocation::Host:
                        std::free(ptr);
                        break;

                    case MemoryLocation::HostPinned:
                    case MemoryLocation::Device:
#ifdef PCR_HAS_CUDA
                        // TODO: CUDA deallocation
#endif
                        break;
                }
            }
        }
        band_buffers_.clear();
    }

    ~Impl() {
        deallocate();
    }
};

// ---------------------------------------------------------------------------
// Grid — Public API
// ---------------------------------------------------------------------------

Grid::~Grid() = default;

std::unique_ptr<Grid> Grid::create(int cols, int rows,
                                    const std::vector<BandDesc>& bands,
                                    MemoryLocation loc) {
    if (cols <= 0 || rows <= 0) {
        return nullptr;
    }

    if (bands.empty()) {
        return nullptr;
    }

    auto grid = std::unique_ptr<Grid>(new Grid());
    grid->impl_ = std::make_unique<Impl>();
    grid->impl_->cols_ = cols;
    grid->impl_->rows_ = rows;
    grid->impl_->bands_ = bands;
    grid->impl_->location_ = loc;

    Status status = grid->impl_->allocate();
    if (!status.ok()) {
        return nullptr;
    }

    return grid;
}

std::unique_ptr<Grid> Grid::create_for_tile(const GridConfig& config,
                                             TileIndex tile,
                                             const std::vector<BandDesc>& bands,
                                             MemoryLocation loc) {
    // Get cell range for this tile
    int col_start, row_start, col_count, row_count;
    config.tile_cell_range(tile, col_start, row_start, col_count, row_count);

    return create(col_count, row_count, bands, loc);
}

int Grid::num_bands() const {
    return impl_ ? static_cast<int>(impl_->bands_.size()) : 0;
}

BandDesc Grid::band_desc(int band_index) const {
    if (!impl_ || band_index < 0 || band_index >= num_bands()) {
        return {};
    }
    return impl_->bands_[band_index];
}

int Grid::band_index(const std::string& name) const {
    if (!impl_) return -1;

    for (size_t i = 0; i < impl_->bands_.size(); ++i) {
        if (impl_->bands_[i].name == name) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void* Grid::band_data(int band_index) {
    if (!impl_ || band_index < 0 || band_index >= num_bands()) {
        return nullptr;
    }
    return impl_->band_buffers_[band_index];
}

const void* Grid::band_data(int band_index) const {
    if (!impl_ || band_index < 0 || band_index >= num_bands()) {
        return nullptr;
    }
    return impl_->band_buffers_[band_index];
}

float* Grid::band_f32(int band_index) {
    return static_cast<float*>(band_data(band_index));
}

const float* Grid::band_f32(int band_index) const {
    return static_cast<const float*>(band_data(band_index));
}

float* Grid::band_f32(const std::string& name) {
    int idx = band_index(name);
    return idx >= 0 ? band_f32(idx) : nullptr;
}

const float* Grid::band_f32(const std::string& name) const {
    int idx = band_index(name);
    return idx >= 0 ? band_f32(idx) : nullptr;
}

int Grid::cols() const {
    return impl_ ? impl_->cols_ : 0;
}

int Grid::rows() const {
    return impl_ ? impl_->rows_ : 0;
}

int64_t Grid::cell_count() const {
    return impl_ ? impl_->cell_count() : 0;
}

MemoryLocation Grid::location() const {
    return impl_ ? impl_->location_ : MemoryLocation::Host;
}

Status Grid::fill(float value) {
    if (!impl_) {
        return Status::error(StatusCode::InvalidArgument, "Grid not initialized");
    }

    for (int i = 0; i < num_bands(); ++i) {
        Status s = fill_band(i, value);
        if (!s.ok()) return s;
    }

    return Status::success();
}

Status Grid::fill_band(int band_index, float value) {
    if (!impl_) {
        return Status::error(StatusCode::InvalidArgument, "Grid not initialized");
    }

    if (band_index < 0 || band_index >= num_bands()) {
        return Status::error(StatusCode::InvalidArgument, "Band index out of range");
    }

    const BandDesc& desc = impl_->bands_[band_index];
    const int64_t cells = impl_->cell_count();
    void* data = impl_->band_buffers_[band_index];

    // Only support Float32 for now (reduction state is always float32)
    if (desc.dtype != DataType::Float32) {
        return Status::error(StatusCode::NotImplemented,
                           "Only Float32 bands supported for fill");
    }

    float* fdata = static_cast<float*>(data);

    // CPU-only path
    if (impl_->location_ != MemoryLocation::Host) {
        return Status::error(StatusCode::NotImplemented,
                           "Fill on non-host memory requires CUDA build");
    }

    std::fill_n(fdata, cells, value);

    return Status::success();
}

std::unique_ptr<Grid> Grid::to(MemoryLocation dst) const {
    if (!impl_) return nullptr;

    // Create destination grid
    auto dst_grid = create(impl_->cols_, impl_->rows_, impl_->bands_, dst);
    if (!dst_grid) return nullptr;

    // Copy data
    Status s = dst_grid->copy_from(*this);
    if (!s.ok()) return nullptr;

    return dst_grid;
}

std::unique_ptr<Grid> Grid::to_device_async(void* cuda_stream) const {
#ifdef PCR_HAS_CUDA
    // TODO: async device transfer with CUDA stream
    return nullptr;
#else
    (void)cuda_stream;  // suppress warning
    return nullptr;
#endif
}

Status Grid::copy_from(const Grid& other, void* cuda_stream) {
    if (!impl_ || !other.impl_) {
        return Status::error(StatusCode::InvalidArgument, "Grid not initialized");
    }

    // Check dimensions match
    if (impl_->cols_ != other.impl_->cols_ || impl_->rows_ != other.impl_->rows_) {
        return Status::error(StatusCode::InvalidArgument,
                           "Grid dimensions do not match");
    }

    // Check band count matches
    if (impl_->bands_.size() != other.impl_->bands_.size()) {
        return Status::error(StatusCode::InvalidArgument,
                           "Band count does not match");
    }

    const int64_t cells = impl_->cell_count();

    // Copy each band
    for (size_t i = 0; i < impl_->bands_.size(); ++i) {
        const BandDesc& desc = impl_->bands_[i];
        const size_t bytes = cells * data_type_size(desc.dtype);

        void* dst_ptr = impl_->band_buffers_[i];
        const void* src_ptr = other.impl_->band_buffers_[i];

        // CPU-only copy for now
        if (impl_->location_ == MemoryLocation::Host &&
            other.impl_->location_ == MemoryLocation::Host) {
            std::memcpy(dst_ptr, src_ptr, bytes);
        } else {
#ifdef PCR_HAS_CUDA
            // TODO: CUDA memcpy (device<->host, device<->device)
            (void)cuda_stream;
            return Status::error(StatusCode::NotImplemented,
                               "CUDA memory copy not yet implemented");
#else
            (void)cuda_stream;
            return Status::error(StatusCode::NotImplemented,
                               "Cross-device copy requires CUDA build");
#endif
        }
    }

    return Status::success();
}

std::vector<uint8_t> Grid::valid_mask(int band_index) const {
    if (!impl_) return {};

    if (band_index < 0 || band_index >= num_bands()) {
        return {};
    }

    const int64_t cells = impl_->cell_count();
    std::vector<uint8_t> mask(cells, 0);

    // Only support Float32 for now
    const BandDesc& desc = impl_->bands_[band_index];
    if (desc.dtype != DataType::Float32) {
        return mask;  // all invalid
    }

    const float* data = band_f32(band_index);
    if (!data) return mask;

    // CPU-only path
    if (impl_->location_ != MemoryLocation::Host) {
        return mask;  // can't read non-host memory
    }

    for (int64_t i = 0; i < cells; ++i) {
        mask[i] = std::isfinite(data[i]) ? 1 : 0;
    }

    return mask;
}

} // namespace pcr
