#include "pcr/core/point_cloud.h"
#include <cstring>
#include <algorithm>
#include <stdexcept>
#ifdef PCR_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace pcr {

// ---------------------------------------------------------------------------
// PointCloud::Impl — PIMPL implementation
// ---------------------------------------------------------------------------
struct PointCloud::Impl {
    // Core coordinate arrays (always Float64 for geo precision)
    double* x_data_ = nullptr;
    double* y_data_ = nullptr;

    // Point count and capacity
    size_t count_ = 0;
    size_t capacity_ = 0;

    // Memory location and ownership
    MemoryLocation location_ = MemoryLocation::Host;
    bool owns_coords_ = true;  // false if wrapped external buffers

    // CRS
    CRS crs_;

    // Named channels (SoA)
    std::unordered_map<std::string, ChannelDesc> channels_;
    std::unordered_map<std::string, void*> channel_buffers_;

    // Allocate coordinate arrays
    Status allocate_coords(size_t cap) {
        capacity_ = cap;
        owns_coords_ = true;

        const size_t bytes = cap * sizeof(double);

        switch (location_) {
            case MemoryLocation::Host:
                x_data_ = static_cast<double*>(std::malloc(bytes));
                y_data_ = static_cast<double*>(std::malloc(bytes));

                if (!x_data_ || !y_data_) {
                    if (x_data_) std::free(x_data_);
                    if (y_data_) std::free(y_data_);
                    x_data_ = y_data_ = nullptr;
                    return Status::error(StatusCode::OutOfMemory,
                                       "Failed to allocate coordinate arrays");
                }
                break;

            case MemoryLocation::HostPinned:
#ifdef PCR_HAS_CUDA
                CUDA_CHECK(cudaMallocHost(&x_data_, bytes));
                CUDA_CHECK(cudaMallocHost(&y_data_, bytes));
                break;
#else
                return Status::error(StatusCode::NotImplemented,
                                   "HostPinned requires CUDA build");
#endif

            case MemoryLocation::Device:
#ifdef PCR_HAS_CUDA
                CUDA_CHECK(cudaMalloc(&x_data_, bytes));
                CUDA_CHECK(cudaMalloc(&y_data_, bytes));
                break;
#else
                return Status::error(StatusCode::NotImplemented,
                                   "Device requires CUDA build");
#endif
        }

        return Status::success();
    }

    // Wrap external coordinate arrays (non-owning)
    void wrap_coords(double* x, double* y, size_t n) {
        x_data_ = x;
        y_data_ = y;
        count_ = n;
        capacity_ = n;
        owns_coords_ = false;
    }

    // Allocate a channel
    Status allocate_channel(const std::string& name, DataType dtype) {
        if (channels_.count(name)) {
            return Status::error(StatusCode::InvalidArgument,
                               "Channel already exists: " + name);
        }

        const size_t elem_size = data_type_size(dtype);
        const size_t bytes = capacity_ * elem_size;

        void* ptr = nullptr;

        switch (location_) {
            case MemoryLocation::Host:
                ptr = std::malloc(bytes);
                if (!ptr) {
                    return Status::error(StatusCode::OutOfMemory,
                                       "Failed to allocate channel: " + name);
                }
                break;

            case MemoryLocation::HostPinned:
#ifdef PCR_HAS_CUDA
                CUDA_CHECK(cudaMallocHost(&ptr, bytes));
                break;
#else
                return Status::error(StatusCode::NotImplemented,
                                   "HostPinned requires CUDA build");
#endif

            case MemoryLocation::Device:
#ifdef PCR_HAS_CUDA
                CUDA_CHECK(cudaMalloc(&ptr, bytes));
                break;
#else
                return Status::error(StatusCode::NotImplemented,
                                   "Device requires CUDA build");
#endif
        }

        // Register channel
        ChannelDesc desc;
        desc.name = name;
        desc.dtype = dtype;
        desc.offset = 0;  // unused in this implementation

        channels_[name] = desc;
        channel_buffers_[name] = ptr;

        return Status::success();
    }

    // Free all memory
    void deallocate() {
        // Free coordinates if owned
        if (owns_coords_) {
            switch (location_) {
                case MemoryLocation::Host:
                    if (x_data_) std::free(x_data_);
                    if (y_data_) std::free(y_data_);
                    break;

                case MemoryLocation::HostPinned:
#ifdef PCR_HAS_CUDA
                    if (x_data_) cudaFreeHost(x_data_);
                    if (y_data_) cudaFreeHost(y_data_);
#endif
                    break;

                case MemoryLocation::Device:
#ifdef PCR_HAS_CUDA
                    if (x_data_) cudaFree(x_data_);
                    if (y_data_) cudaFree(y_data_);
#endif
                    break;
            }
        }

        x_data_ = y_data_ = nullptr;
        owns_coords_ = false;

        // Free channels
        for (auto& [name, ptr] : channel_buffers_) {
            if (ptr) {
                switch (location_) {
                    case MemoryLocation::Host:
                        std::free(ptr);
                        break;

                    case MemoryLocation::HostPinned:
#ifdef PCR_HAS_CUDA
                        cudaFreeHost(ptr);
#endif
                        break;

                    case MemoryLocation::Device:
#ifdef PCR_HAS_CUDA
                        cudaFree(ptr);
#endif
                        break;
                }
            }
        }

        channel_buffers_.clear();
        channels_.clear();
    }

    ~Impl() {
        deallocate();
    }
};

// ---------------------------------------------------------------------------
// PointCloud — Public API
// ---------------------------------------------------------------------------

PointCloud::~PointCloud() = default;

std::unique_ptr<PointCloud> PointCloud::create(size_t capacity,
                                                MemoryLocation loc) {
    if (capacity == 0) {
        return nullptr;
    }

    auto cloud = std::unique_ptr<PointCloud>(new PointCloud());
    cloud->impl_ = std::make_unique<Impl>();
    cloud->impl_->location_ = loc;

    Status status = cloud->impl_->allocate_coords(capacity);
    if (!status.ok()) {
        return nullptr;
    }

    return cloud;
}

std::unique_ptr<PointCloud> PointCloud::wrap(double* x, double* y, size_t count,
                                              MemoryLocation loc) {
    if (!x || !y || count == 0) {
        return nullptr;
    }

    auto cloud = std::unique_ptr<PointCloud>(new PointCloud());
    cloud->impl_ = std::make_unique<Impl>();
    cloud->impl_->location_ = loc;
    cloud->impl_->wrap_coords(x, y, count);

    return cloud;
}

Status PointCloud::add_channel(const std::string& name, DataType dtype) {
    if (!impl_) {
        return Status::error(StatusCode::InvalidArgument, "PointCloud not initialized");
    }

    return impl_->allocate_channel(name, dtype);
}

bool PointCloud::has_channel(const std::string& name) const {
    if (!impl_) return false;
    return impl_->channels_.count(name) > 0;
}

const ChannelDesc* PointCloud::channel(const std::string& name) const {
    if (!impl_) return nullptr;

    auto it = impl_->channels_.find(name);
    if (it == impl_->channels_.end()) {
        return nullptr;
    }

    return &it->second;
}

std::vector<std::string> PointCloud::channel_names() const {
    if (!impl_) return {};

    std::vector<std::string> names;
    names.reserve(impl_->channels_.size());

    for (const auto& [name, desc] : impl_->channels_) {
        names.push_back(name);
    }

    return names;
}

double* PointCloud::x() {
    return impl_ ? impl_->x_data_ : nullptr;
}

const double* PointCloud::x() const {
    return impl_ ? impl_->x_data_ : nullptr;
}

double* PointCloud::y() {
    return impl_ ? impl_->y_data_ : nullptr;
}

const double* PointCloud::y() const {
    return impl_ ? impl_->y_data_ : nullptr;
}

void* PointCloud::channel_data(const std::string& name) {
    if (!impl_) return nullptr;

    auto it = impl_->channel_buffers_.find(name);
    if (it == impl_->channel_buffers_.end()) {
        return nullptr;
    }

    return it->second;
}

const void* PointCloud::channel_data(const std::string& name) const {
    if (!impl_) return nullptr;

    auto it = impl_->channel_buffers_.find(name);
    if (it == impl_->channel_buffers_.end()) {
        return nullptr;
    }

    return it->second;
}

float* PointCloud::channel_f32(const std::string& name) {
    const ChannelDesc* desc = channel(name);
    if (!desc || desc->dtype != DataType::Float32) {
        return nullptr;
    }
    return static_cast<float*>(channel_data(name));
}

const float* PointCloud::channel_f32(const std::string& name) const {
    const ChannelDesc* desc = channel(name);
    if (!desc || desc->dtype != DataType::Float32) {
        return nullptr;
    }
    return static_cast<const float*>(channel_data(name));
}

int32_t* PointCloud::channel_i32(const std::string& name) {
    const ChannelDesc* desc = channel(name);
    if (!desc || desc->dtype != DataType::Int32) {
        return nullptr;
    }
    return static_cast<int32_t*>(channel_data(name));
}

const int32_t* PointCloud::channel_i32(const std::string& name) const {
    const ChannelDesc* desc = channel(name);
    if (!desc || desc->dtype != DataType::Int32) {
        return nullptr;
    }
    return static_cast<const int32_t*>(channel_data(name));
}

size_t PointCloud::count() const {
    return impl_ ? impl_->count_ : 0;
}

size_t PointCloud::capacity() const {
    return impl_ ? impl_->capacity_ : 0;
}

MemoryLocation PointCloud::location() const {
    return impl_ ? impl_->location_ : MemoryLocation::Host;
}

CRS PointCloud::crs() const {
    return impl_ ? impl_->crs_ : CRS{};
}

void PointCloud::set_crs(const CRS& crs) {
    if (impl_) {
        impl_->crs_ = crs;
    }
}

Status PointCloud::resize(size_t new_count) {
    if (!impl_) {
        return Status::error(StatusCode::InvalidArgument, "PointCloud not initialized");
    }

    if (new_count > impl_->capacity_) {
        return Status::error(StatusCode::InvalidArgument,
                           "Cannot resize beyond capacity. Use create() with larger capacity.");
    }

    impl_->count_ = new_count;
    return Status::success();
}

std::unique_ptr<PointCloud> PointCloud::to(MemoryLocation dst) const {
    if (!impl_) {
        fprintf(stderr, "PointCloud::to() - impl_ is null\n");
        return nullptr;
    }

    // Create destination cloud
    auto dst_cloud = create(impl_->capacity_, dst);
    if (!dst_cloud) {
        fprintf(stderr, "PointCloud::to() - Failed to create destination cloud "
                "(capacity=%zu, location=%d)\n", impl_->capacity_, static_cast<int>(dst));
        return nullptr;
    }

    // Set count and CRS
    dst_cloud->impl_->count_ = impl_->count_;
    dst_cloud->impl_->crs_ = impl_->crs_;

    // Copy coordinates
    const size_t coord_bytes = impl_->count_ * sizeof(double);

    if (impl_->location_ == MemoryLocation::Host && dst == MemoryLocation::Host) {
        std::memcpy(dst_cloud->impl_->x_data_, impl_->x_data_, coord_bytes);
        std::memcpy(dst_cloud->impl_->y_data_, impl_->y_data_, coord_bytes);
    } else {
#ifdef PCR_HAS_CUDA
        // Use cudaMemcpyDefault for automatic direction detection
        cudaError_t err;
        err = cudaMemcpy(dst_cloud->impl_->x_data_, impl_->x_data_, coord_bytes, cudaMemcpyDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "PointCloud::to() - cudaMemcpy failed for X coordinates: %s\n",
                    cudaGetErrorString(err));
            return nullptr;
        }

        err = cudaMemcpy(dst_cloud->impl_->y_data_, impl_->y_data_, coord_bytes, cudaMemcpyDefault);
        if (err != cudaSuccess) {
            fprintf(stderr, "PointCloud::to() - cudaMemcpy failed for Y coordinates: %s\n",
                    cudaGetErrorString(err));
            return nullptr;
        }
#else
        fprintf(stderr, "PointCloud::to() - CUDA not available\n");
        return nullptr;
#endif
    }

    // Copy channels
    for (const auto& [name, desc] : impl_->channels_) {
        // Allocate channel in destination
        Status s = dst_cloud->add_channel(name, desc.dtype);
        if (!s.ok()) {
            return nullptr;
        }

        // Copy data
        const size_t elem_size = data_type_size(desc.dtype);
        const size_t bytes = impl_->count_ * elem_size;

        void* src_ptr = impl_->channel_buffers_.at(name);
        void* dst_ptr = dst_cloud->impl_->channel_buffers_.at(name);

        if (impl_->location_ == MemoryLocation::Host && dst == MemoryLocation::Host) {
            std::memcpy(dst_ptr, src_ptr, bytes);
        } else {
#ifdef PCR_HAS_CUDA
            cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, bytes, cudaMemcpyDefault);
            if (err != cudaSuccess) {
                fprintf(stderr, "PointCloud::to() - cudaMemcpy failed for channel '%s': %s\n",
                        name.c_str(), cudaGetErrorString(err));
                return nullptr;
            }
#else
            fprintf(stderr, "PointCloud::to() - CUDA not available for channel copy\n");
            return nullptr;
#endif
        }
    }

    return dst_cloud;
}

std::unique_ptr<PointCloud> PointCloud::to_device_async(void* cuda_stream) const {
#ifdef PCR_HAS_CUDA
    if (!impl_) return nullptr;

    // Create destination cloud on Device
    auto dst_cloud = create(impl_->capacity_, MemoryLocation::Device);
    if (!dst_cloud) return nullptr;

    // Set count and CRS
    dst_cloud->impl_->count_ = impl_->count_;
    dst_cloud->impl_->crs_ = impl_->crs_;

    cudaStream_t stream = static_cast<cudaStream_t>(cuda_stream);

    // Async copy coordinates
    const size_t coord_bytes = impl_->count_ * sizeof(double);
    cudaError_t err;

    err = cudaMemcpyAsync(dst_cloud->impl_->x_data_, impl_->x_data_,
                          coord_bytes, cudaMemcpyDefault, stream);
    if (err != cudaSuccess) return nullptr;

    err = cudaMemcpyAsync(dst_cloud->impl_->y_data_, impl_->y_data_,
                          coord_bytes, cudaMemcpyDefault, stream);
    if (err != cudaSuccess) return nullptr;

    // Async copy channels
    for (const auto& [name, desc] : impl_->channels_) {
        // Allocate channel in destination
        Status s = dst_cloud->add_channel(name, desc.dtype);
        if (!s.ok()) return nullptr;

        // Async copy data
        const size_t elem_size = data_type_size(desc.dtype);
        const size_t bytes = impl_->count_ * elem_size;

        void* src_ptr = impl_->channel_buffers_.at(name);
        void* dst_ptr = dst_cloud->impl_->channel_buffers_.at(name);

        err = cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDefault, stream);
        if (err != cudaSuccess) return nullptr;
    }

    return dst_cloud;
#else
    (void)cuda_stream;
    return nullptr;
#endif
}

} // namespace pcr
