# ==============================================================
# Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2024.1 (64-bit)
# Tool Version Limit: 2024.05
# Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
# Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
# 
# ==============================================================
cmake_minimum_required(VERSION 3.15)
project(conv2d_kernel)
find_package(common)
collector_create (PROJECT_LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}")
collect (PROJECT_LIB_SOURCES xconv2d_kernel.c)
collect (PROJECT_LIB_SOURCES xconv2d_kernel_sinit.c)
collect (PROJECT_LIB_SOURCES xconv2d_kernel_linux.c)
collect (PROJECT_LIB_SOURCES xconv2d_kernel_g.c)
collector_list (_sources PROJECT_LIB_SOURCES)
collector_create (PROJECT_LIB_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}")
collect (PROJECT_LIB_HEADERS xconv2d_kernel.h)
collect (PROJECT_LIB_HEADERS xconv2d_kernel_hw.h)
collector_list (_headers PROJECT_LIB_HEADERS)
file(COPY ${_headers} DESTINATION ${CMAKE_BINARY_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)
add_library(conv2d_kernel STATIC ${_sources})
set_target_properties(conv2d_kernel PROPERTIES LINKER_LANGUAGE C)

