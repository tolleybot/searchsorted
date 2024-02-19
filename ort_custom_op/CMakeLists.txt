cmake_minimum_required(VERSION 3.10)
project(ortsearchsortedop)
enable_language(CUDA)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

find_package(onnxruntime)

# Find the Microsoft GSL package
find_package(Microsoft.GSL CONFIG REQUIRED)


set(TORCH_ROOT_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../torch")
message(STATUS "TORCH_ROOT_SRC: ${TORCH_ROOT_SRC}")

include_directories(${onnxruntime_INCLUDE_DIRS}/onnxruntime/include ${TORCH_ROOT_SRC}/cuda)

set(CUDA_SOURCE_FILES "${TORCH_ROOT_SRC}/cuda/searchsorted_cuda_kernel.cu" ort_searchsorted_op.cu) 
set(CUDA_HEADER_FILES "${TORCH_ROOT_SRC}/cuda/searchsorted_cuda_kernel.h")

#set(TEST_SOURCE testsearchsorted.cpp)
set(HEADER ort_searchsorted_op.h)
set(SOURCE registration.cpp)
#add_executable(ortsearchsortedop ${SOURCE} ${HEADER} ${TEST_SOURCE})
# Create a shared library target
add_library(ortsearchsortedop SHARED ${SOURCE} ${HEADER} ${CUDA_SOURCE_FILES})
 
# Use target-specific include directories
target_include_directories(ortsearchsortedop PRIVATE
    "/onnxruntime/onnxruntime"
    ${onnxruntime_INCLUDE_DIRS}
    ${onnxruntime_INCLUDE_DIRS}/core/session
    ${TORCH_ROOT_SRC}/cuda
)


target_link_libraries(ortsearchsortedop PUBLIC ${onnxruntime_LIBRARY} Microsoft.GSL::GSL)