cmake_minimum_required(VERSION 3.18)
project(PCR VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# CMake module path
# ---------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(PCR_ENABLE_CUDA    "Enable CUDA GPU acceleration"    OFF)
option(PCR_BUILD_TESTS    "Build C++ tests"                ON)
option(PCR_BUILD_PYTHON   "Build Python bindings"          ON)
option(PCR_BUILD_EXAMPLES "Build example programs"         ON)
option(PCR_BUILD_BENCHMARKS "Build CUDA benchmarks"        OFF)

# ---------------------------------------------------------------------------
# C++ standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# CUDA setup (optional)
# ---------------------------------------------------------------------------
if(PCR_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Define macro for conditional compilation
    add_compile_definitions(PCR_HAS_CUDA)

    # CUDA architecture (compute capability 7.0+ = Volta, Turing, Ampere, etc.)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
    endif()

    message(STATUS "CUDA enabled - targeting architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

# OpenMP for CPU multi-threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled for CPU multi-threading")
    add_compile_definitions(PCR_HAS_OPENMP)
else()
    message(STATUS "OpenMP not found - CPU code will run single-threaded")
endif()

# ---------------------------------------------------------------------------
# Find dependencies
# ---------------------------------------------------------------------------

# GDAL (GeoTIFF I/O)
find_package(GDAL REQUIRED)
if(GDAL_FOUND)
    message(STATUS "Found GDAL: ${GDAL_LIBRARY}")
endif()

# PROJ (CRS transformations)
find_package(PROJ REQUIRED)
if(PROJ_FOUND)
    message(STATUS "Found PROJ: ${PROJ_LIBRARY}")
endif()

# GoogleTest (for C++ tests)
if(PCR_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        message(STATUS "GTest not found, will fetch from source")
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
        )
        FetchContent_MakeAvailable(googletest)
    endif()
endif()

# pybind11 (for Python bindings)
if(PCR_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        message(STATUS "pybind11 not found, will fetch from source")
        include(FetchContent)
        FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1
        )
        FetchContent_MakeAvailable(pybind11)
    endif()
endif()

# ---------------------------------------------------------------------------
# Include directories
# ---------------------------------------------------------------------------
include_directories(${CMAKE_SOURCE_DIR}/include)

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------

# Collect source files
set(PCR_CORE_SOURCES
    src/core/types.cpp
    src/core/grid_config.cpp
    src/core/grid.cpp
    src/core/point_cloud.cpp
)

set(PCR_OPS_SOURCES
    src/ops/reduction_registry.cpp
)

set(PCR_IO_SOURCES
    src/io/tile_state_io.cpp
    src/io/grid_io.cpp
    src/io/point_cloud_io.cpp
)

set(PCR_ENGINE_SOURCES
    src/engine/tile_manager.cpp
    src/engine/pipeline.cpp
    src/engine/reprojection.cpp
    src/engine/accumulator.cpp
    src/engine/filter.cpp
    src/engine/tile_router.cpp
)

# CUDA sources (only if CUDA enabled)
set(PCR_ENGINE_CUDA_SOURCES
    src/engine/memory_pool.cu
    src/engine/accumulator_kernels.cu
    src/engine/filter_kernels.cu
    src/engine/tile_router_kernels.cu
    src/engine/grid_merge.cu
)

# Combine all sources
set(PCR_SOURCES
    ${PCR_CORE_SOURCES}
    ${PCR_OPS_SOURCES}
    ${PCR_IO_SOURCES}
    ${PCR_ENGINE_SOURCES}
)

# Add CUDA sources if enabled
if(PCR_ENABLE_CUDA)
    list(APPEND PCR_SOURCES ${PCR_ENGINE_CUDA_SOURCES})
endif()

# Create library
add_library(pcr ${PCR_SOURCES})

# Link dependencies
target_link_libraries(pcr PUBLIC
    GDAL::GDAL
    PROJ::proj
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(pcr PUBLIC OpenMP::OpenMP_CXX)
endif()

# Include directories for library consumers
target_include_directories(pcr PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Add CUDA include directories if CUDA is enabled
# This is needed for .cpp files that include CUDA headers
if(PCR_ENABLE_CUDA)
    target_include_directories(pcr PRIVATE
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
endif()

# Position-independent code (required for Python bindings)
set_target_properties(pcr PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(MSVC)
    target_compile_options(pcr PRIVATE /W4)
else()
    target_compile_options(pcr PRIVATE -Wall -Wextra -pedantic)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PCR_BUILD_TESTS)
    enable_testing()

    set(PCR_TEST_SOURCES
        tests/cpp/test_types.cpp
        tests/cpp/test_grid_config.cpp
        tests/cpp/test_grid.cpp
        tests/cpp/test_point_cloud.cpp
        tests/cpp/test_reduction_ops.cpp
        tests/cpp/test_tile_state_io.cpp
        tests/cpp/test_grid_io.cpp
        tests/cpp/test_point_cloud_io.cpp
        tests/cpp/test_filter.cpp
        tests/cpp/test_tile_router.cpp
        tests/cpp/test_accumulator.cpp
        tests/cpp/test_tile_manager.cpp
        tests/cpp/test_pipeline.cpp
        tests/cpp/test_error_handling.cpp
    )

    # GPU tests (only if CUDA enabled)
    if(PCR_ENABLE_CUDA)
        list(APPEND PCR_TEST_SOURCES
            tests/cpp/test_gpu.cpp
            tests/cpp/test_gpu_pipeline.cpp
        )
    endif()

    # Threading tests (only if OpenMP enabled)
    if(OpenMP_CXX_FOUND)
        list(APPEND PCR_TEST_SOURCES
            tests/cpp/test_threading.cpp
        )
    endif()

    # CUDA tests (only if CUDA enabled)
    # TODO: Implement dedicated CUDA test files
    # if(PCR_ENABLE_CUDA)
    #     list(APPEND PCR_TEST_SOURCES
    #         tests/cpp/test_accumulator.cu
    #         tests/cpp/test_filter.cu
    #         tests/cpp/test_grid_merge.cu
    #     )
    # endif()

    foreach(test_source ${PCR_TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE pcr GTest::gtest_main)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
if(PCR_BUILD_EXAMPLES)
    set(PCR_EXAMPLES
        examples/cpp/basic_rasterize.cpp
        examples/cpp/multi_collection.cpp
        examples/cpp/tiled_large_grid.cpp
    )

    foreach(example_source ${PCR_EXAMPLES})
        get_filename_component(example_name ${example_source} NAME_WE)
        add_executable(${example_name} ${example_source})
        target_link_libraries(${example_name} PRIVATE pcr)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Benchmarks (CUDA only)
# ---------------------------------------------------------------------------
if(PCR_BUILD_BENCHMARKS AND PCR_ENABLE_CUDA)
    set(PCR_BENCHMARKS
        benchmarks/bench_compare_cpu_gpu.cu
    )

    foreach(bench_source ${PCR_BENCHMARKS})
        get_filename_component(bench_name ${bench_source} NAME_WE)
        add_executable(${bench_name} ${bench_source})
        target_link_libraries(${bench_name} PRIVATE pcr)
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Python bindings
# ---------------------------------------------------------------------------
if(PCR_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install(TARGETS pcr
    EXPORT PCRTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/pcr
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT PCRTargets
    FILE PCRTargets.cmake
    NAMESPACE PCR::
    DESTINATION lib/cmake/PCR
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
message(STATUS "")
message(STATUS "PCR Configuration Summary:")
message(STATUS "  CUDA enabled:      ${PCR_ENABLE_CUDA}")
message(STATUS "  Build tests:       ${PCR_BUILD_TESTS}")
message(STATUS "  Build Python:      ${PCR_BUILD_PYTHON}")
message(STATUS "  Build examples:    ${PCR_BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks:  ${PCR_BUILD_BENCHMARKS}")
message(STATUS "")
