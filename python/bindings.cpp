#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "pcr/core/types.h"
#include "pcr/core/grid_config.h"
#include "pcr/core/grid.h"
#include "pcr/core/point_cloud.h"
#include "pcr/engine/filter.h"
#include "pcr/engine/pipeline.h"
#include "pcr/io/grid_io.h"
#include "pcr/io/point_cloud_io.h"

namespace py = pybind11;
using namespace pcr;

// ---------------------------------------------------------------------------
// Helper: Convert Status to Python exception
// ---------------------------------------------------------------------------
void check_status(const Status& s) {
    if (!s.ok()) {
        throw std::runtime_error(s.message);
    }
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_pcr, m) {
    m.doc() = "Point Cloud Rasterization library";

    // -----------------------------------------------------------------------
    // Enums
    // -----------------------------------------------------------------------
    py::enum_<DataType>(m, "DataType")
        .value("Float32", DataType::Float32)
        .value("Float64", DataType::Float64)
        .value("Int32", DataType::Int32)
        .value("UInt32", DataType::UInt32)
        .value("Int16", DataType::Int16)
        .value("UInt16", DataType::UInt16)
        .value("UInt8", DataType::UInt8)
        .export_values();

    py::enum_<ReductionType>(m, "ReductionType")
        .value("Sum", ReductionType::Sum)
        .value("Max", ReductionType::Max)
        .value("Min", ReductionType::Min)
        .value("Average", ReductionType::Average)
        .value("WeightedAverage", ReductionType::WeightedAverage)
        .value("Count", ReductionType::Count)
        .value("Median", ReductionType::Median)
        .value("Percentile", ReductionType::Percentile)
        .value("MostRecent", ReductionType::MostRecent)
        .value("PriorityMerge", ReductionType::PriorityMerge)
        .value("Custom", ReductionType::Custom)
        .export_values();

    py::enum_<MemoryLocation>(m, "MemoryLocation")
        .value("Host", MemoryLocation::Host)
        .value("HostPinned", MemoryLocation::HostPinned)
        .value("Device", MemoryLocation::Device)
        .export_values();

    py::enum_<ExecutionMode>(m, "ExecutionMode")
        .value("CPU", ExecutionMode::CPU)
        .value("GPU", ExecutionMode::GPU)
        .value("Auto", ExecutionMode::Auto)
        .value("Hybrid", ExecutionMode::Hybrid)
        .export_values();

    py::enum_<StatusCode>(m, "StatusCode")
        .value("Ok", StatusCode::Ok)
        .value("InvalidArgument", StatusCode::InvalidArgument)
        .value("OutOfMemory", StatusCode::OutOfMemory)
        .value("CudaError", StatusCode::CudaError)
        .value("IoError", StatusCode::IoError)
        .value("CrsError", StatusCode::CrsError)
        .value("NotImplemented", StatusCode::NotImplemented)
        .export_values();

    py::enum_<CompareOp>(m, "CompareOp")
        .value("Equal", CompareOp::Equal)
        .value("NotEqual", CompareOp::NotEqual)
        .value("Less", CompareOp::Less)
        .value("LessEqual", CompareOp::LessEqual)
        .value("Greater", CompareOp::Greater)
        .value("GreaterEqual", CompareOp::GreaterEqual)
        .value("InSet", CompareOp::InSet)
        .value("NotInSet", CompareOp::NotInSet)
        .export_values();

    py::enum_<PointCloudFormat>(m, "PointCloudFormat")
        .value("PCR_Binary", PointCloudFormat::PCR_Binary)
        .value("CSV", PointCloudFormat::CSV)
        .value("LAS", PointCloudFormat::LAS)
        .value("LAZ", PointCloudFormat::LAZ)
        .value("Auto", PointCloudFormat::Auto)
        .export_values();

    // -----------------------------------------------------------------------
    // Core types
    // -----------------------------------------------------------------------
    py::class_<BBox>(m, "BBox")
        .def(py::init<>())
        .def_readwrite("min_x", &BBox::min_x)
        .def_readwrite("min_y", &BBox::min_y)
        .def_readwrite("max_x", &BBox::max_x)
        .def_readwrite("max_y", &BBox::max_y)
        .def("expand", py::overload_cast<double, double>(&BBox::expand))
        .def("expand", py::overload_cast<const BBox&>(&BBox::expand))
        .def("contains", &BBox::contains)
        .def("width", &BBox::width)
        .def("height", &BBox::height)
        .def("valid", &BBox::valid)
        .def("__repr__", [](const BBox& b) {
            return "BBox(min_x=" + std::to_string(b.min_x) +
                   ", min_y=" + std::to_string(b.min_y) +
                   ", max_x=" + std::to_string(b.max_x) +
                   ", max_y=" + std::to_string(b.max_y) + ")";
        });

    py::class_<CRS>(m, "CRS")
        .def(py::init<>())
        .def_readwrite("wkt", &CRS::wkt)
        .def_readwrite("epsg", &CRS::epsg)
        .def("is_projected", &CRS::is_projected)
        .def("is_geographic", &CRS::is_geographic)
        .def("is_valid", &CRS::is_valid)
        .def_static("from_epsg", &CRS::from_epsg)
        .def_static("from_wkt", &CRS::from_wkt)
        .def("equivalent_to", &CRS::equivalent_to)
        .def("__repr__", [](const CRS& c) {
            if (c.epsg != 0) {
                return "CRS(epsg=" + std::to_string(c.epsg) + ")";
            }
            return "CRS(wkt='" + c.wkt.substr(0, 50) + "...')";
        });

    py::class_<NoDataPolicy>(m, "NoDataPolicy")
        .def(py::init<>())
        .def_readwrite("value", &NoDataPolicy::value)
        .def_readwrite("use_nan", &NoDataPolicy::use_nan)
        .def("sentinel", &NoDataPolicy::sentinel);

    py::class_<TileIndex>(m, "TileIndex")
        .def(py::init<>())
        .def(py::init<int, int>())
        .def_readwrite("row", &TileIndex::row)
        .def_readwrite("col", &TileIndex::col)
        .def("__eq__", &TileIndex::operator==)
        .def("__lt__", &TileIndex::operator<)
        .def("__repr__", [](const TileIndex& t) {
            return "TileIndex(row=" + std::to_string(t.row) +
                   ", col=" + std::to_string(t.col) + ")";
        });

    py::class_<Status>(m, "Status")
        .def(py::init<>())
        .def_readwrite("code", &Status::code)
        .def_readwrite("message", &Status::message)
        .def("ok", &Status::ok)
        .def_static("success", &Status::success)
        .def_static("error", &Status::error)
        .def("__bool__", &Status::ok)
        .def("__repr__", [](const Status& s) -> std::string {
            if (s.ok()) return "Status(Ok)";
            return "Status(code=" + std::to_string(static_cast<int>(s.code)) +
                   ", message='" + s.message + "')";
        });

    py::class_<ChannelDesc>(m, "ChannelDesc")
        .def(py::init<>())
        .def_readwrite("name", &ChannelDesc::name)
        .def_readwrite("dtype", &ChannelDesc::dtype)
        .def_readwrite("offset", &ChannelDesc::offset);

    py::class_<BandDesc>(m, "BandDesc")
        .def(py::init<>())
        .def_readwrite("name", &BandDesc::name)
        .def_readwrite("dtype", &BandDesc::dtype)
        .def_readwrite("is_state", &BandDesc::is_state);

    // -----------------------------------------------------------------------
    // GridConfig
    // -----------------------------------------------------------------------
    py::class_<GridConfig>(m, "GridConfig")
        .def(py::init<>())
        .def_readwrite("bounds", &GridConfig::bounds)
        .def_readwrite("crs", &GridConfig::crs)
        .def_readwrite("cell_size_x", &GridConfig::cell_size_x)
        .def_readwrite("cell_size_y", &GridConfig::cell_size_y)
        .def_readwrite("width", &GridConfig::width)
        .def_readwrite("height", &GridConfig::height)
        .def_readwrite("nodata", &GridConfig::nodata)
        .def_readwrite("tile_width", &GridConfig::tile_width)
        .def_readwrite("tile_height", &GridConfig::tile_height)
        .def_readwrite("tiles_x", &GridConfig::tiles_x)
        .def_readwrite("tiles_y", &GridConfig::tiles_y)
        .def("compute_dimensions", &GridConfig::compute_dimensions)
        .def("world_to_cell", [](const GridConfig& cfg, double wx, double wy) {
            int col = 0, row = 0;
            bool valid = cfg.world_to_cell(wx, wy, col, row);
            return py::make_tuple(col, row, valid);
        })
        .def("cell_to_world", [](const GridConfig& cfg, int col, int row) {
            double wx = 0.0, wy = 0.0;
            cfg.cell_to_world(col, row, wx, wy);
            return py::make_tuple(wx, wy);
        })
        .def("cell_to_tile", &GridConfig::cell_to_tile)
        .def("tile_bounds", &GridConfig::tile_bounds)
        .def("tile_cell_range", [](const GridConfig& cfg, TileIndex idx) {
            int col_start = 0, row_start = 0, col_count = 0, row_count = 0;
            cfg.tile_cell_range(idx, col_start, row_start, col_count, row_count);
            return py::make_tuple(col_start, row_start, col_count, row_count);
        })
        .def("total_tiles", &GridConfig::total_tiles)
        .def("total_cells", &GridConfig::total_cells)
        .def("validate", [](const GridConfig& cfg) {
            check_status(cfg.validate());
        })
        .def("__repr__", [](const GridConfig& cfg) {
            return "GridConfig(width=" + std::to_string(cfg.width) +
                   ", height=" + std::to_string(cfg.height) +
                   ", tiles=" + std::to_string(cfg.tiles_x) + "x" + std::to_string(cfg.tiles_y) + ")";
        });

    // -----------------------------------------------------------------------
    // Grid
    // -----------------------------------------------------------------------
    py::class_<Grid>(m, "Grid")
        .def_static("create", &Grid::create,
            py::arg("cols"), py::arg("rows"),
            py::arg("bands"),
            py::arg("loc") = MemoryLocation::Host)
        .def_static("create_for_tile", &Grid::create_for_tile,
            py::arg("config"), py::arg("tile"),
            py::arg("bands"),
            py::arg("loc") = MemoryLocation::Host)
        .def("num_bands", &Grid::num_bands)
        .def("band_desc", &Grid::band_desc)
        .def("band_index", &Grid::band_index)
        .def("cols", &Grid::cols)
        .def("rows", &Grid::rows)
        .def("cell_count", &Grid::cell_count)
        .def("location", &Grid::location)
        .def("fill", [](Grid& g, float value) {
            check_status(g.fill(value));
        })
        .def("fill_band", [](Grid& g, int band_index, float value) {
            check_status(g.fill_band(band_index, value));
        })
        // NumPy array access
        .def("band_array", [](Grid& g, int band_index) {
            float* data = g.band_f32(band_index);
            if (!data) {
                throw std::runtime_error("Invalid band index or data type");
            }
            return py::array_t<float>(
                {g.rows(), g.cols()},
                {g.cols() * sizeof(float), sizeof(float)},
                data,
                py::cast(&g)  // Keep Grid alive while array exists
            );
        })
        .def("set_band_array", [](Grid& g, int band_index, py::array_t<float> arr) {
            float* data = g.band_f32(band_index);
            if (!data) {
                throw std::runtime_error("Invalid band index or data type");
            }
            auto buf = arr.request();
            if (buf.shape[0] != g.rows() || buf.shape[1] != g.cols()) {
                throw std::runtime_error("Array shape mismatch");
            }
            std::memcpy(data, buf.ptr, g.cell_count() * sizeof(float));
        })
        .def("__repr__", [](const Grid& g) {
            return "Grid(cols=" + std::to_string(g.cols()) +
                   ", rows=" + std::to_string(g.rows()) +
                   ", bands=" + std::to_string(g.num_bands()) + ")";
        });

    // -----------------------------------------------------------------------
    // PointCloud
    // -----------------------------------------------------------------------
    py::class_<PointCloud>(m, "PointCloud")
        .def_static("create", &PointCloud::create,
            py::arg("capacity"),
            py::arg("loc") = MemoryLocation::Host)
        .def("add_channel", [](PointCloud& pc, const std::string& name, DataType dtype) {
            check_status(pc.add_channel(name, dtype));
        })
        .def("has_channel", &PointCloud::has_channel)
        .def("channel", &PointCloud::channel, py::return_value_policy::reference_internal)
        .def("channel_names", &PointCloud::channel_names)
        .def("count", &PointCloud::count)
        .def("capacity", &PointCloud::capacity)
        .def("location", &PointCloud::location)
        .def("crs", &PointCloud::crs)
        .def("set_crs", &PointCloud::set_crs)
        .def("resize", [](PointCloud& pc, size_t new_count) {
            check_status(pc.resize(new_count));
        })
        // NumPy array access for coordinates
        .def("x_array", [](PointCloud& pc) {
            double* data = pc.x();
            return py::array_t<double>(
                {pc.count()},
                {sizeof(double)},
                data,
                py::cast(&pc)
            );
        })
        .def("y_array", [](PointCloud& pc) {
            double* data = pc.y();
            return py::array_t<double>(
                {pc.count()},
                {sizeof(double)},
                data,
                py::cast(&pc)
            );
        })
        .def("channel_array_f32", [](PointCloud& pc, const std::string& name) {
            float* data = pc.channel_f32(name);
            if (!data) {
                throw std::runtime_error("Channel not found or wrong type: " + name);
            }
            return py::array_t<float>(
                {pc.count()},
                {sizeof(float)},
                data,
                py::cast(&pc)
            );
        })
        .def("set_x_array", [](PointCloud& pc, py::array_t<double> arr) {
            double* data = pc.x();
            auto buf = arr.request();
            if (buf.shape[0] > static_cast<ssize_t>(pc.capacity())) {
                throw std::runtime_error("Array too large for capacity");
            }
            std::memcpy(data, buf.ptr, buf.shape[0] * sizeof(double));
            check_status(pc.resize(buf.shape[0]));
        })
        .def("set_y_array", [](PointCloud& pc, py::array_t<double> arr) {
            double* data = pc.y();
            auto buf = arr.request();
            if (buf.shape[0] > static_cast<ssize_t>(pc.capacity())) {
                throw std::runtime_error("Array too large for capacity");
            }
            std::memcpy(data, buf.ptr, buf.shape[0] * sizeof(double));
        })
        .def("set_channel_array_f32", [](PointCloud& pc, const std::string& name, py::array_t<float> arr) {
            float* data = pc.channel_f32(name);
            if (!data) {
                throw std::runtime_error("Channel not found or wrong type: " + name);
            }
            auto buf = arr.request();
            if (buf.shape[0] > static_cast<ssize_t>(pc.count())) {
                throw std::runtime_error("Array size exceeds point count");
            }
            std::memcpy(data, buf.ptr, buf.shape[0] * sizeof(float));
        })
        // Device memory transfer
        .def("to_device", [](const PointCloud& pc) {
            auto result = pc.to(MemoryLocation::Device);
            if (!result) {
                throw std::runtime_error(
                    "Failed to transfer point cloud to Device memory. "
                    "Possible causes: CUDA out of memory, CUDA not initialized, "
                    "or incompatible GPU configuration.");
            }
            return result;
        }, "Transfer point cloud to GPU Device memory")
        .def("to_host", [](const PointCloud& pc) {
            auto result = pc.to(MemoryLocation::Host);
            if (!result) {
                throw std::runtime_error("Failed to transfer point cloud to Host memory");
            }
            return result;
        }, "Transfer point cloud to Host memory")
        .def("__repr__", [](const PointCloud& pc) {
            return "PointCloud(count=" + std::to_string(pc.count()) +
                   ", capacity=" + std::to_string(pc.capacity()) +
                   ", channels=" + std::to_string(pc.channel_names().size()) + ")";
        });

    // -----------------------------------------------------------------------
    // Filter
    // -----------------------------------------------------------------------
    py::class_<FilterPredicate>(m, "FilterPredicate")
        .def(py::init<>())
        .def_readwrite("channel_name", &FilterPredicate::channel_name)
        .def_readwrite("op", &FilterPredicate::op)
        .def_readwrite("value", &FilterPredicate::value)
        .def_readwrite("value_set", &FilterPredicate::value_set);

    py::class_<FilterSpec>(m, "FilterSpec")
        .def(py::init<>())
        .def_readwrite("predicates", &FilterSpec::predicates)
        .def("add", &FilterSpec::add,
            py::arg("channel"), py::arg("op"), py::arg("value"),
            py::return_value_policy::reference_internal)
        .def("add_in_set", &FilterSpec::add_in_set,
            py::arg("channel"), py::arg("values"),
            py::return_value_policy::reference_internal)
        .def("empty", &FilterSpec::empty);

    // -----------------------------------------------------------------------
    // Pipeline
    // -----------------------------------------------------------------------
    py::class_<ReductionSpec>(m, "ReductionSpec")
        .def(py::init<>())
        .def_readwrite("value_channel", &ReductionSpec::value_channel)
        .def_readwrite("type", &ReductionSpec::type)
        .def_readwrite("weight_channel", &ReductionSpec::weight_channel)
        .def_readwrite("timestamp_channel", &ReductionSpec::timestamp_channel)
        .def_readwrite("percentile", &ReductionSpec::percentile)
        .def_readwrite("output_band_name", &ReductionSpec::output_band_name);

    py::class_<PipelineConfig>(m, "PipelineConfig")
        .def(py::init<>())
        .def_readwrite("grid", &PipelineConfig::grid)
        .def_readwrite("reductions", &PipelineConfig::reductions)
        .def_readwrite("filter", &PipelineConfig::filter)
        .def_readwrite("target_crs", &PipelineConfig::target_crs)
        .def_readwrite("auto_reproject", &PipelineConfig::auto_reproject)
        .def_readwrite("exec_mode", &PipelineConfig::exec_mode)
        .def_readwrite("gpu_memory_budget", &PipelineConfig::gpu_memory_budget)
        .def_readwrite("host_cache_budget", &PipelineConfig::host_cache_budget)
        .def_readwrite("chunk_size", &PipelineConfig::chunk_size)
        .def_readwrite("cpu_threads", &PipelineConfig::cpu_threads)
        .def_readwrite("gpu_fallback_to_cpu", &PipelineConfig::gpu_fallback_to_cpu)
        .def_readwrite("hybrid_cpu_threads", &PipelineConfig::hybrid_cpu_threads)
        .def_readwrite("state_dir", &PipelineConfig::state_dir)
        .def_readwrite("resume", &PipelineConfig::resume)
        .def_readwrite("output_path", &PipelineConfig::output_path)
        .def_readwrite("write_cog", &PipelineConfig::write_cog);

    py::class_<ProgressInfo>(m, "ProgressInfo")
        .def(py::init<>())
        .def_readwrite("collections_processed", &ProgressInfo::collections_processed)
        .def_readwrite("collections_total", &ProgressInfo::collections_total)
        .def_readwrite("points_processed", &ProgressInfo::points_processed)
        .def_readwrite("tiles_active", &ProgressInfo::tiles_active)
        .def_readwrite("elapsed_seconds", &ProgressInfo::elapsed_seconds)
        .def("__repr__", [](const ProgressInfo& info) {
            return "ProgressInfo(points=" + std::to_string(info.points_processed) +
                   ", tiles=" + std::to_string(info.tiles_active) +
                   ", elapsed=" + std::to_string(info.elapsed_seconds) + "s)";
        });

    py::class_<Pipeline>(m, "Pipeline")
        .def_static("create", &Pipeline::create)
        .def("validate", [](const Pipeline& p) {
            check_status(p.validate());
        })
        .def("ingest", [](Pipeline& p, const PointCloud& cloud) {
            check_status(p.ingest(cloud));
        })
        .def("finalize", [](Pipeline& p) {
            check_status(p.finalize());
        })
        .def("run", [](Pipeline& p, const std::vector<const PointCloud*>& clouds) {
            check_status(p.run(clouds));
        })
        .def("set_progress_callback", &Pipeline::set_progress_callback)
        .def("result", &Pipeline::result, py::return_value_policy::reference_internal)
        .def("stats", &Pipeline::stats);

    // -----------------------------------------------------------------------
    // I/O - GeoTIFF
    // -----------------------------------------------------------------------
    py::class_<GeoTiffOptions>(m, "GeoTiffOptions")
        .def(py::init<>())
        .def_readwrite("cloud_optimized", &GeoTiffOptions::cloud_optimized)
        .def_readwrite("compress", &GeoTiffOptions::compress)
        .def_readwrite("compress_level", &GeoTiffOptions::compress_level)
        .def_readwrite("tile_width", &GeoTiffOptions::tile_width)
        .def_readwrite("tile_height", &GeoTiffOptions::tile_height)
        .def_readwrite("bigtiff", &GeoTiffOptions::bigtiff)
        .def_readwrite("overview_resampling", &GeoTiffOptions::overview_resampling);

    m.def("write_geotiff", [](const std::string& path, const Grid& grid,
                              const GridConfig& config, const GeoTiffOptions& options) {
        check_status(write_geotiff(path, grid, config, options));
    }, py::arg("path"), py::arg("grid"), py::arg("config"),
       py::arg("options") = GeoTiffOptions());

    m.def("read_geotiff_info", [](const std::string& path) {
        int width, height, num_bands;
        CRS crs;
        BBox bounds;
        check_status(read_geotiff_info(path, width, height, num_bands, crs, bounds));
        return py::make_tuple(width, height, num_bands, crs, bounds);
    });

    // -----------------------------------------------------------------------
    // I/O - Point Cloud
    // -----------------------------------------------------------------------
    py::class_<PointCloudInfo>(m, "PointCloudInfo")
        .def(py::init<>())
        .def_readwrite("num_points", &PointCloudInfo::num_points)
        .def_readwrite("channels", &PointCloudInfo::channels)
        .def_readwrite("crs", &PointCloudInfo::crs)
        .def_readwrite("bounds", &PointCloudInfo::bounds);

    m.def("read_point_cloud", [](const std::string& path, PointCloudFormat format) {
        auto cloud = read_point_cloud(path, format);
        if (!cloud) {
            throw std::runtime_error("Failed to read point cloud: " + path);
        }
        return cloud;
    }, py::arg("path"), py::arg("format") = PointCloudFormat::Auto);

    m.def("write_point_cloud", [](const std::string& path, const PointCloud& cloud,
                                   PointCloudFormat format) {
        check_status(write_point_cloud(path, cloud, format));
    }, py::arg("path"), py::arg("cloud"),
       py::arg("format") = PointCloudFormat::PCR_Binary);

    m.def("read_point_cloud_info", [](const std::string& path, PointCloudFormat format) {
        PointCloudInfo info;
        check_status(read_point_cloud_info(path, info, format));
        return info;
    }, py::arg("path"), py::arg("format") = PointCloudFormat::Auto);

    py::class_<PointCloudReader>(m, "PointCloudReader")
        .def_static("open", [](const std::string& path, PointCloudFormat format) {
            auto reader = PointCloudReader::open(path, format);
            if (!reader) {
                throw std::runtime_error("Failed to open point cloud: " + path);
            }
            return reader;
        }, py::arg("path"), py::arg("format") = PointCloudFormat::Auto)
        .def("info", &PointCloudReader::info, py::return_value_policy::reference_internal)
        .def("read_chunk", &PointCloudReader::read_chunk)
        .def("rewind", [](PointCloudReader& r) {
            check_status(r.rewind());
        })
        .def("eof", &PointCloudReader::eof);
}
