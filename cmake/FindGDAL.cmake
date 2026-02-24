# FindGDAL.cmake - Locate GDAL library
#
# Sets:
#   GDAL_FOUND - System has GDAL
#   GDAL_INCLUDE_DIR - GDAL include directory
#   GDAL_LIBRARY - GDAL library
#   GDAL::GDAL - Imported target

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_GDAL QUIET gdal)
endif()

find_path(GDAL_INCLUDE_DIR
    NAMES gdal.h
    HINTS
        ${PC_GDAL_INCLUDE_DIRS}
    PATHS
        /usr/include/gdal
        /usr/local/include/gdal
        /usr/include
        /usr/local/include
)

find_library(GDAL_LIBRARY
    NAMES gdal gdal_i
    HINTS
        ${PC_GDAL_LIBRARY_DIRS}
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GDAL
    REQUIRED_VARS GDAL_LIBRARY GDAL_INCLUDE_DIR
)

if(GDAL_FOUND AND NOT TARGET GDAL::GDAL)
    add_library(GDAL::GDAL UNKNOWN IMPORTED)
    set_target_properties(GDAL::GDAL PROPERTIES
        IMPORTED_LOCATION "${GDAL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${GDAL_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(GDAL_INCLUDE_DIR GDAL_LIBRARY)
