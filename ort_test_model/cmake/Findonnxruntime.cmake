# Custom cmake config file by jcarius to enable find_package(onnxruntime) without modifying LIBRARY_PATH and LD_LIBRARY_PATH
#
# This will define the following variables:
#   onnxruntime_FOUND        -- True if the system has the onnxruntime library
#   onnxruntime_INCLUDE_DIRS -- The include directories for onnxruntime
#   onnxruntime_LIBRARIES    -- Libraries to link against
#   onnxruntime_CXX_FLAGS    -- Additional (required) compiler flags

include(FindPackageHandleStandardArgs)

# Correctly setting the installation prefix
get_filename_component(onnxruntime_INSTALL_PREFIX "/usr/local" ABSOLUTE)
set(onnxruntime_INCLUDE_DIRS ${onnxruntime_INSTALL_PREFIX}/include/onnxruntime/)
set(onnxruntime_LIBRARIES onnxruntime)
set(onnxruntime_CXX_FLAGS "") # no flags needed

message(STATUS "ONNX Runtime installation prefix set to: ${onnxruntime_INSTALL_PREFIX}")

# Assuming the library is in the standard lib directory under the installation prefix
find_library(onnxruntime_LIBRARY onnxruntime
    PATHS "${onnxruntime_INSTALL_PREFIX}/lib"
    NO_DEFAULT_PATH) # This ensures it only looks in the specified directory

# Now configure the imported target
add_library(onnxruntime SHARED IMPORTED)
set_property(TARGET onnxruntime PROPERTY IMPORTED_LOCATION "${onnxruntime_LIBRARY}")
set_property(TARGET onnxruntime PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${onnxruntime_INCLUDE_DIRS}")

# Check if ONNX Runtime was found successfully
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(onnxruntime DEFAULT_MSG onnxruntime_LIBRARY onnxruntime_INCLUDE_DIRS)
 
message(STATUS "ONNX Runtime include directories set to: ${onnxruntime_INCLUDE_DIRS}")
message(STATUS "ONNX Runtime library found at: ${onnxruntime_LIBRARY}")

if(onnxruntime_FOUND)
    message(STATUS "ONNX Runtime found successfully.")
else()
    message(WARNING "ONNX Runtime not found.")
endif()



