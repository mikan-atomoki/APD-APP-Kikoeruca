cmake_minimum_required(VERSION 3.22)
project(apd_engine LANGUAGES CXX)

add_library(apd_engine SHARED
    apd_kernels.cpp
    apd_inference.cpp
    jni_bridge.cpp
)

target_include_directories(apd_engine PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(apd_engine PRIVATE -O3 -ffast-math -march=armv8-a+simd)
target_link_libraries(apd_engine android log)
