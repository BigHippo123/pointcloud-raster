# FindPROJ.cmake - Locate PROJ library
#
# Sets:
#   PROJ_FOUND - System has PROJ
#   PROJ_INCLUDE_DIR - PROJ include directory
#   PROJ_LIBRARY - PROJ library
#   PROJ::proj - Imported target

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(PC_PROJ QUIET proj)
endif()

find_path(PROJ_INCLUDE_DIR
    NAMES proj.h
    HINTS
        ${PC_PROJ_INCLUDE_DIRS}
        /usr/include
        /usr/local/include
)

find_library(PROJ_LIBRARY
    NAMES proj proj_i
    HINTS
        ${PC_PROJ_LIBRARY_DIRS}
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PROJ
    REQUIRED_VARS PROJ_LIBRARY PROJ_INCLUDE_DIR
)

if(PROJ_FOUND AND NOT TARGET PROJ::proj)
    add_library(PROJ::proj UNKNOWN IMPORTED)
    set_target_properties(PROJ::proj PROPERTIES
        IMPORTED_LOCATION "${PROJ_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${PROJ_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(PROJ_INCLUDE_DIR PROJ_LIBRARY)
