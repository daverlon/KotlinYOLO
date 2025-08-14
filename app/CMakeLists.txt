cmake_minimum_required(VERSION 3.18)

project("yolo_compute_offload")

add_library(
        yolo_compute_offload
        SHARED
        src/main/cpp/yolo_preprocess.cpp
        src/main/cpp/yolo_postprocess.cpp
)

# Find JNI
#find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Set C++ standard
set_target_properties(yolo_compute_offload PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
)

if(CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
    target_compile_options(yolo_compute_offload PRIVATE -mfpu=neon -O3)
elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    target_compile_options(yolo_compute_offload PRIVATE -O3)
endif()