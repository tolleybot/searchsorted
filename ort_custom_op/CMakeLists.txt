cmake_minimum_required(VERSION 3.10)
project(ortsearchsortedop)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

find_package(onnxruntime)

# Find the Microsoft GSL package
find_package(Microsoft.GSL CONFIG REQUIRED)


include_directories(${onnxruntime_INCLUDE_DIRS}/onnxruntime/include)


#set(TEST_SOURCE testsearchsorted.cpp)
set(HEADER ort_searchsorted_op.h)
set(SOURCE ort_searchsorted_op.cpp)
#add_executable(ortsearchsortedop ${SOURCE} ${HEADER} ${TEST_SOURCE})
# Create a shared library target
add_library(ortsearchsortedop SHARED ${SOURCE} ${HEADER})

#Include path to header files for Custom Op
include_directories(${onnxruntime_INCLUDE_DIRS} "/onnxruntime/onnxruntime")

target_link_libraries(ortsearchsortedop PUBLIC ${onnxruntime_LIBRARY} Microsoft.GSL::GSL)