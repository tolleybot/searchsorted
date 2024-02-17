cmake_minimum_required(VERSION 3.10)
project(ortsearchsortedop)


if(NOT ONNXRUNTIME_ROOTDIR)
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
endif()

set(HEADER ort_searchsorted_op.h)
set(SOURCE ort_searchsorted_op.cpp)
add_executable(customop ${SOURCE} ${HEADER} ${TEST_SOURCE})

#Include path to header files for Custom Op
include_directories("${ONNXRUNTIME_ROOTDIR}/include/onnxruntime")
#Linking dependencies for Custom Op
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS "/usr/local/lib" REQUIRED)

target_link_libraries(ortsearchsortedop PUBLIC ${ONNXRUNTIME_LIBRARY})