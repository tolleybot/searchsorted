cmake_minimum_required(VERSION 3.13)  # Adjust minimum version as necessary
project(onnx_inference)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# Find ONNX Runtime package 
find_package(onnxruntime REQUIRED)

# Add executable
add_executable(onnx_inference main.cpp)  # Replace 'main.cpp' with the name of your source file

# Link against ONNX Runtime. Adjust the library name if you're using a GPU or other specific build.
target_link_libraries(onnx_inference PRIVATE onnxruntime)

# Optional: If you have additional include directories, link them here
target_include_directories(onnx_inference PRIVATE 
${onnxruntime_INCLUDE_DIRS}/onnxruntime/include
${onnxruntime_INCLUDE_DIRS}/core/session/)
