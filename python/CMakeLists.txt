# Python bindings using pybind11

# Create Python extension module
pybind11_add_module(_pcr bindings.cpp)

# Link against the PCR library
target_link_libraries(_pcr PRIVATE pcr)

# Set output directory to python/pcr so it can be imported
set_target_properties(_pcr PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/pcr"
)

# Install the module
install(TARGETS _pcr
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/python/pcr
)

# Install Python package files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/pcr
    DESTINATION ${CMAKE_INSTALL_PREFIX}/python
    FILES_MATCHING PATTERN "*.py"
)
