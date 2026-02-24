#include "pcr/io/point_cloud_io.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <sys/stat.h>

namespace pcr {

// ===========================================================================
// Constants and Helpers
// ===========================================================================

static constexpr uint32_t MAGIC_PCRP = 0x50524350;  // "PCRP" in little-endian
static constexpr uint32_t FORMAT_VERSION = 1;

namespace {

bool ends_with(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) return false;
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

PointCloudFormat detect_format(const std::string& path) {
    std::string lower_path = path;
    std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);

    if (ends_with(lower_path, ".pcrp")) return PointCloudFormat::PCR_Binary;
    if (ends_with(lower_path, ".csv")) return PointCloudFormat::CSV;
    if (ends_with(lower_path, ".las")) return PointCloudFormat::LAS;
    if (ends_with(lower_path, ".laz")) return PointCloudFormat::LAZ;

    // Try to detect by reading magic number
    std::ifstream file(path, std::ios::binary);
    if (file) {
        uint32_t magic;
        file.read(reinterpret_cast<char*>(&magic), 4);
        if (magic == MAGIC_PCRP) {
            return PointCloudFormat::PCR_Binary;
        }
    }

    // Default to CSV
    return PointCloudFormat::CSV;
}

uint8_t data_type_to_byte(DataType dt) {
    return static_cast<uint8_t>(dt);
}

DataType byte_to_data_type(uint8_t b) {
    return static_cast<DataType>(b);
}

size_t count_csv_lines(const std::string& path) {
    std::ifstream file(path);
    if (!file) return 0;

    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++count;
    }
    return count > 0 ? count - 1 : 0;  // subtract header
}

} // anonymous namespace

// ===========================================================================
// PCR Binary Format Implementation
// ===========================================================================

namespace {

Status write_pcr_binary(const std::string& path, const PointCloud& cloud) {
    if (cloud.location() != MemoryLocation::Host) {
        return Status::error(StatusCode::InvalidArgument, "cloud must be on host");
    }

    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return Status::error(StatusCode::IoError, "failed to open file for writing: " + path);
    }

    // Write header
    uint32_t magic = MAGIC_PCRP;
    uint32_t version = FORMAT_VERSION;
    uint64_t num_points = cloud.count();
    auto channel_names = cloud.channel_names();
    uint32_t num_channels = channel_names.size();

    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&num_points), sizeof(num_points));
    ofs.write(reinterpret_cast<const char*>(&num_channels), sizeof(num_channels));

    // Write CRS
    std::string crs_wkt = cloud.crs().wkt;
    uint32_t crs_wkt_len = crs_wkt.size();
    ofs.write(reinterpret_cast<const char*>(&crs_wkt_len), sizeof(crs_wkt_len));
    if (crs_wkt_len > 0) {
        ofs.write(crs_wkt.c_str(), crs_wkt_len);
    }

    // Write channel table
    for (const auto& name : channel_names) {
        const ChannelDesc* desc = cloud.channel(name);
        if (!desc) continue;
        uint16_t name_len = name.size();
        uint8_t dtype = data_type_to_byte(desc->dtype);

        ofs.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
        ofs.write(name.c_str(), name_len);
        ofs.write(reinterpret_cast<const char*>(&dtype), sizeof(dtype));
    }

    if (!ofs) {
        return Status::error(StatusCode::IoError, "failed to write header");
    }

    // Write body (SoA)
    // X coordinates
    ofs.write(reinterpret_cast<const char*>(cloud.x()), num_points * sizeof(double));

    // Y coordinates
    ofs.write(reinterpret_cast<const char*>(cloud.y()), num_points * sizeof(double));

    // Channels
    for (const auto& name : channel_names) {
        const ChannelDesc* desc = cloud.channel(name);
        if (!desc) continue;
        size_t elem_size = data_type_size(desc->dtype);

        const void* data_ptr = cloud.channel_data(name);
        if (!data_ptr) {
            return Status::error(StatusCode::InvalidArgument,
                "failed to get channel data: " + name);
        }

        ofs.write(reinterpret_cast<const char*>(data_ptr), num_points * elem_size);
    }

    if (!ofs) {
        return Status::error(StatusCode::IoError, "failed to write point data");
    }

    ofs.close();
    return Status::success();
}

Status read_pcr_binary_info(const std::string& path, PointCloudInfo& info) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to open file: " + path);
    }

    // Read header
    uint32_t magic, version;
    uint64_t num_points;
    uint32_t num_channels;

    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    ifs.read(reinterpret_cast<char*>(&num_points), sizeof(num_points));
    ifs.read(reinterpret_cast<char*>(&num_channels), sizeof(num_channels));

    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to read header");
    }

    if (magic != MAGIC_PCRP) {
        return Status::error(StatusCode::IoError, "invalid magic number (not a PCRP file)");
    }

    if (version != FORMAT_VERSION) {
        return Status::error(StatusCode::IoError,
            "unsupported version " + std::to_string(version));
    }

    // Read CRS
    uint32_t crs_wkt_len;
    ifs.read(reinterpret_cast<char*>(&crs_wkt_len), sizeof(crs_wkt_len));

    std::string crs_wkt;
    if (crs_wkt_len > 0) {
        crs_wkt.resize(crs_wkt_len);
        ifs.read(&crs_wkt[0], crs_wkt_len);
    }

    // Read channel table
    std::vector<ChannelDesc> channels;
    for (uint32_t i = 0; i < num_channels; ++i) {
        uint16_t name_len;
        ifs.read(reinterpret_cast<char*>(&name_len), sizeof(name_len));

        std::string name(name_len, '\0');
        ifs.read(&name[0], name_len);

        uint8_t dtype_byte;
        ifs.read(reinterpret_cast<char*>(&dtype_byte), sizeof(dtype_byte));

        ChannelDesc desc;
        desc.name = name;
        desc.dtype = byte_to_data_type(dtype_byte);
        channels.push_back(desc);
    }

    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to read channel table");
    }

    // Fill info
    info.num_points = num_points;
    info.channels = channels;
    info.crs.wkt = crs_wkt;
    info.bounds = BBox();  // Not stored in format

    return Status::success();
}

std::unique_ptr<PointCloud> read_pcr_binary(const std::string& path) {
    PointCloudInfo info;
    Status s = read_pcr_binary_info(path, info);
    if (!s.ok()) {
        return nullptr;
    }

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return nullptr;
    }

    // Skip to body
    // Header size calculation
    size_t header_size = 4 + 4 + 8 + 4;  // magic, version, num_points, num_channels
    header_size += 4 + info.crs.wkt.size();  // crs_wkt_len + wkt
    for (const auto& ch : info.channels) {
        header_size += 2 + ch.name.size() + 1;  // name_len + name + dtype
    }

    ifs.seekg(header_size, std::ios::beg);
    if (!ifs) {
        return nullptr;
    }

    // Create point cloud
    auto cloud = PointCloud::create(info.num_points, MemoryLocation::Host);
    cloud->resize(info.num_points);  // Set count
    cloud->set_crs(info.crs);

    // Read X
    std::vector<double> x_temp(info.num_points);
    ifs.read(reinterpret_cast<char*>(x_temp.data()), info.num_points * sizeof(double));

    // Read Y
    std::vector<double> y_temp(info.num_points);
    ifs.read(reinterpret_cast<char*>(y_temp.data()), info.num_points * sizeof(double));

    // Copy coordinates
    memcpy(const_cast<double*>(cloud->x()), x_temp.data(), info.num_points * sizeof(double));
    memcpy(const_cast<double*>(cloud->y()), y_temp.data(), info.num_points * sizeof(double));

    // Read channels
    for (const auto& ch : info.channels) {
        cloud->add_channel(ch.name, ch.dtype);
        size_t elem_size = data_type_size(ch.dtype);

        std::vector<uint8_t> temp_data(info.num_points * elem_size);
        ifs.read(reinterpret_cast<char*>(temp_data.data()), info.num_points * elem_size);

        void* channel_ptr = cloud->channel_data(ch.name);
        if (channel_ptr) {
            memcpy(channel_ptr, temp_data.data(), info.num_points * elem_size);
        }
    }

    if (!ifs) {
        return nullptr;
    }

    return cloud;
}

} // anonymous namespace

// ===========================================================================
// CSV Format Implementation
// ===========================================================================

namespace {

Status write_csv(const std::string& path, const PointCloud& cloud) {
    if (cloud.location() != MemoryLocation::Host) {
        return Status::error(StatusCode::InvalidArgument, "cloud must be on host");
    }

    std::ofstream ofs(path);
    if (!ofs) {
        return Status::error(StatusCode::IoError, "failed to open file for writing: " + path);
    }

    ofs << std::setprecision(15);

    // Write header
    ofs << "x,y";
    auto channel_names = cloud.channel_names();
    for (const auto& name : channel_names) {
        ofs << "," << name;
    }
    ofs << "\n";

    // Write data rows
    for (size_t i = 0; i < cloud.count(); ++i) {
        ofs << cloud.x()[i] << "," << cloud.y()[i];

        for (const auto& name : channel_names) {
            const ChannelDesc* desc = cloud.channel(name);
            if (!desc) continue;
            ofs << ",";

            switch (desc->dtype) {
                case DataType::Float32:
                    ofs << cloud.channel_f32(name)[i];
                    break;
                case DataType::Float64: {
                    const double* data = static_cast<const double*>(cloud.channel_data(name));
                    ofs << data[i];
                    break;
                }
                case DataType::Int32:
                    ofs << cloud.channel_i32(name)[i];
                    break;
                case DataType::UInt32: {
                    const uint32_t* data = static_cast<const uint32_t*>(cloud.channel_data(name));
                    ofs << data[i];
                    break;
                }
                default:
                    return Status::error(StatusCode::InvalidArgument,
                        "unsupported channel data type");
            }
        }
        ofs << "\n";
    }

    if (!ofs) {
        return Status::error(StatusCode::IoError, "failed to write CSV data");
    }

    return Status::success();
}

Status read_csv_info(const std::string& path, PointCloudInfo& info) {
    std::ifstream ifs(path);
    if (!ifs) {
        return Status::error(StatusCode::IoError, "failed to open file: " + path);
    }

    // Read header line
    std::string header_line;
    if (!std::getline(ifs, header_line)) {
        return Status::error(StatusCode::IoError, "empty CSV file");
    }

    // Parse header
    std::vector<std::string> headers;
    std::istringstream header_stream(header_line);
    std::string token;
    while (std::getline(header_stream, token, ',')) {
        headers.push_back(token);
    }

    if (headers.size() < 2 || headers[0] != "x" || headers[1] != "y") {
        return Status::error(StatusCode::IoError, "CSV must start with x,y columns");
    }

    // Create channel descriptors (assume Float64 for all)
    std::vector<ChannelDesc> channels;
    for (size_t i = 2; i < headers.size(); ++i) {
        ChannelDesc desc;
        desc.name = headers[i];
        desc.dtype = DataType::Float64;
        channels.push_back(desc);
    }

    // Count lines
    size_t num_points = count_csv_lines(path);

    info.num_points = num_points;
    info.channels = channels;
    info.crs = CRS();
    info.bounds = BBox();

    return Status::success();
}

std::unique_ptr<PointCloud> read_csv(const std::string& path) {
    PointCloudInfo info;
    Status s = read_csv_info(path, info);
    if (!s.ok()) {
        return nullptr;
    }

    std::ifstream ifs(path);
    if (!ifs) {
        return nullptr;
    }

    // Skip header
    std::string header_line;
    std::getline(ifs, header_line);

    // Create point cloud
    auto cloud = PointCloud::create(info.num_points, MemoryLocation::Host);
    cloud->resize(info.num_points);  // Set count
    cloud->set_crs(info.crs);

    // Add channels
    for (const auto& ch : info.channels) {
        cloud->add_channel(ch.name, ch.dtype);
    }

    // Read data
    std::string line;
    size_t idx = 0;

    std::vector<double> x_temp(info.num_points);
    std::vector<double> y_temp(info.num_points);
    std::vector<std::vector<double>> channel_data(info.channels.size());
    for (auto& vec : channel_data) {
        vec.resize(info.num_points);
    }

    while (std::getline(ifs, line) && idx < info.num_points) {
        std::istringstream line_stream(line);
        std::string token;

        // Read x
        if (!std::getline(line_stream, token, ',')) break;
        x_temp[idx] = std::stod(token);

        // Read y
        if (!std::getline(line_stream, token, ',')) break;
        y_temp[idx] = std::stod(token);

        // Read channels
        for (size_t ch_idx = 0; ch_idx < info.channels.size(); ++ch_idx) {
            if (!std::getline(line_stream, token, ',')) break;
            channel_data[ch_idx][idx] = std::stod(token);
        }

        ++idx;
    }

    // Copy to cloud
    memcpy(const_cast<double*>(cloud->x()), x_temp.data(), idx * sizeof(double));
    memcpy(const_cast<double*>(cloud->y()), y_temp.data(), idx * sizeof(double));

    for (size_t ch_idx = 0; ch_idx < info.channels.size(); ++ch_idx) {
        const auto& ch = info.channels[ch_idx];
        void* channel_ptr = cloud->channel_data(ch.name);
        if (channel_ptr && ch.dtype == DataType::Float64) {
            memcpy(channel_ptr, channel_data[ch_idx].data(), idx * sizeof(double));
        }
    }

    if (idx < info.num_points) {
        cloud->resize(idx);
    }

    return cloud;
}

} // anonymous namespace

// ===========================================================================
// LAS/LAZ Stubs
// ===========================================================================

namespace {

std::unique_ptr<PointCloud> read_las(const std::string& /*path*/) {
    // TODO: Implement LAS reading via PDAL or LASlib
    return nullptr;
}

Status write_las(const std::string& /*path*/, const PointCloud& /*cloud*/) {
    return Status::error(StatusCode::NotImplemented,
        "LAS/LAZ format support not yet implemented");
}

} // anonymous namespace

// ===========================================================================
// High-Level API
// ===========================================================================

std::unique_ptr<PointCloud> read_point_cloud(const std::string& path,
                                             PointCloudFormat format) {
    if (format == PointCloudFormat::Auto) {
        format = detect_format(path);
    }

    switch (format) {
        case PointCloudFormat::PCR_Binary:
            return read_pcr_binary(path);
        case PointCloudFormat::CSV:
            return read_csv(path);
        case PointCloudFormat::LAS:
        case PointCloudFormat::LAZ:
            return read_las(path);
        default:
            return nullptr;
    }
}

Status read_point_cloud_info(const std::string& path,
                             PointCloudInfo& info,
                             PointCloudFormat format) {
    if (format == PointCloudFormat::Auto) {
        format = detect_format(path);
    }

    switch (format) {
        case PointCloudFormat::PCR_Binary:
            return read_pcr_binary_info(path, info);
        case PointCloudFormat::CSV:
            return read_csv_info(path, info);
        case PointCloudFormat::LAS:
        case PointCloudFormat::LAZ:
            return Status::error(StatusCode::NotImplemented,
                "LAS/LAZ format support not yet implemented");
        default:
            return Status::error(StatusCode::InvalidArgument, "unknown format");
    }
}

Status write_point_cloud(const std::string& path,
                        const PointCloud& cloud,
                        PointCloudFormat format) {
    if (format == PointCloudFormat::Auto) {
        format = detect_format(path);
    }

    switch (format) {
        case PointCloudFormat::PCR_Binary:
            return write_pcr_binary(path, cloud);
        case PointCloudFormat::CSV:
            return write_csv(path, cloud);
        case PointCloudFormat::LAS:
        case PointCloudFormat::LAZ:
            return write_las(path, cloud);
        default:
            return Status::error(StatusCode::InvalidArgument, "unknown format");
    }
}

// ===========================================================================
// PointCloudReader Implementation
// ===========================================================================

struct PointCloudReader::Impl {
    std::ifstream file;
    PointCloudInfo info;
    PointCloudFormat format;
    size_t points_read = 0;
    size_t header_offset = 0;  // For PCR Binary

    size_t read_chunk_pcr(PointCloud& cloud, size_t max_points);
    size_t read_chunk_csv(PointCloud& cloud, size_t max_points);
};

size_t PointCloudReader::Impl::read_chunk_pcr(PointCloud& cloud, size_t max_points) {
    if (points_read >= info.num_points) {
        return 0;
    }

    size_t to_read = std::min(max_points, info.num_points - points_read);

    // Read X chunk
    std::vector<double> x_temp(to_read);
    file.read(reinterpret_cast<char*>(x_temp.data()), to_read * sizeof(double));

    // Read Y chunk
    std::vector<double> y_temp(to_read);
    file.read(reinterpret_cast<char*>(y_temp.data()), to_read * sizeof(double));

    // Copy to cloud
    cloud.resize(to_read);
    memcpy(const_cast<double*>(cloud.x()), x_temp.data(), to_read * sizeof(double));
    memcpy(const_cast<double*>(cloud.y()), y_temp.data(), to_read * sizeof(double));

    // Read channels
    for (const auto& ch : info.channels) {
        if (!cloud.has_channel(ch.name)) {
            cloud.add_channel(ch.name, ch.dtype);
        }

        size_t elem_size = data_type_size(ch.dtype);
        std::vector<uint8_t> temp_data(to_read * elem_size);
        file.read(reinterpret_cast<char*>(temp_data.data()), to_read * elem_size);

        void* channel_ptr = cloud.channel_data(ch.name);
        if (channel_ptr) {
            memcpy(channel_ptr, temp_data.data(), to_read * elem_size);
        }
    }

    points_read += to_read;
    return to_read;
}

size_t PointCloudReader::Impl::read_chunk_csv(PointCloud& cloud, size_t max_points) {
    size_t count = 0;
    std::string line;

    cloud.resize(max_points);

    // Ensure channels exist
    for (const auto& ch : info.channels) {
        if (!cloud.has_channel(ch.name)) {
            cloud.add_channel(ch.name, ch.dtype);
        }
    }

    std::vector<double> x_temp, y_temp;
    std::vector<std::vector<double>> channel_data(info.channels.size());

    while (count < max_points && std::getline(file, line)) {
        std::istringstream line_stream(line);
        std::string token;

        // Read x
        if (!std::getline(line_stream, token, ',')) break;
        x_temp.push_back(std::stod(token));

        // Read y
        if (!std::getline(line_stream, token, ',')) break;
        y_temp.push_back(std::stod(token));

        // Read channels
        for (size_t ch_idx = 0; ch_idx < info.channels.size(); ++ch_idx) {
            if (!std::getline(line_stream, token, ',')) break;
            channel_data[ch_idx].push_back(std::stod(token));
        }

        ++count;
        ++points_read;
    }

    // Copy to cloud
    cloud.resize(count);
    if (count > 0) {
        memcpy(const_cast<double*>(cloud.x()), x_temp.data(), count * sizeof(double));
        memcpy(const_cast<double*>(cloud.y()), y_temp.data(), count * sizeof(double));

        for (size_t ch_idx = 0; ch_idx < info.channels.size(); ++ch_idx) {
            const auto& ch = info.channels[ch_idx];
            void* channel_ptr = cloud.channel_data(ch.name);
            if (channel_ptr && ch.dtype == DataType::Float64) {
                memcpy(channel_ptr, channel_data[ch_idx].data(), count * sizeof(double));
            }
        }
    }

    return count;
}

PointCloudReader::~PointCloudReader() = default;

std::unique_ptr<PointCloudReader> PointCloudReader::open(
    const std::string& path,
    PointCloudFormat format)
{
    if (format == PointCloudFormat::Auto) {
        format = detect_format(path);
    }

    auto reader = std::unique_ptr<PointCloudReader>(new PointCloudReader());
    reader->impl_ = std::make_unique<Impl>();
    reader->impl_->format = format;

    // Read info
    Status s = read_point_cloud_info(path, reader->impl_->info, format);
    if (!s.ok()) {
        return nullptr;
    }

    // Open file
    if (format == PointCloudFormat::CSV) {
        reader->impl_->file.open(path);
        if (!reader->impl_->file) {
            return nullptr;
        }
        // Skip header
        std::string header;
        std::getline(reader->impl_->file, header);
    } else if (format == PointCloudFormat::PCR_Binary) {
        reader->impl_->file.open(path, std::ios::binary);
        if (!reader->impl_->file) {
            return nullptr;
        }

        // Calculate header size and skip to body
        size_t header_size = 4 + 4 + 8 + 4;
        header_size += 4 + reader->impl_->info.crs.wkt.size();
        for (const auto& ch : reader->impl_->info.channels) {
            header_size += 2 + ch.name.size() + 1;
        }
        reader->impl_->header_offset = header_size;
        reader->impl_->file.seekg(header_size, std::ios::beg);
    } else {
        return nullptr;
    }

    return reader;
}

const PointCloudInfo& PointCloudReader::info() const {
    return impl_->info;
}

size_t PointCloudReader::read_chunk(PointCloud& cloud, size_t max_points) {
    if (!impl_ || !impl_->file) {
        return 0;
    }

    switch (impl_->format) {
        case PointCloudFormat::PCR_Binary:
            return impl_->read_chunk_pcr(cloud, max_points);
        case PointCloudFormat::CSV:
            return impl_->read_chunk_csv(cloud, max_points);
        default:
            return 0;
    }
}

Status PointCloudReader::rewind() {
    if (!impl_ || !impl_->file) {
        return Status::error(StatusCode::InvalidArgument, "reader not open");
    }

    impl_->points_read = 0;

    if (impl_->format == PointCloudFormat::PCR_Binary) {
        impl_->file.clear();
        impl_->file.seekg(impl_->header_offset, std::ios::beg);
    } else if (impl_->format == PointCloudFormat::CSV) {
        impl_->file.clear();
        impl_->file.seekg(0, std::ios::beg);
        // Skip header
        std::string header;
        std::getline(impl_->file, header);
    }

    return Status::success();
}

bool PointCloudReader::eof() const {
    if (!impl_) return true;
    return impl_->points_read >= impl_->info.num_points;
}

} // namespace pcr
