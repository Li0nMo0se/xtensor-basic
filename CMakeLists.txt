cmake_minimum_required(VERSION 3.18..3.20)

project(mymodule)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)
find_package(tbb REQUIRED)
# Implicit find_package xsmid with xtensor::use_simd

add_definitions(-DXTENSOR_USE_TBB)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math -mavx2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Wextra")

pybind11_add_module(mymodule main.cpp)
# /!\ Crash with C++20 and xt::evaluation_strategy::immediate
set_property(TARGET mymodule PROPERTY CXX_STANDARD 17)
# /!\ Notice that xtensor::use_simd option prevents building on a machine and distributing
# the resulting binary on another machine with a different architecture
target_link_libraries(mymodule PUBLIC pybind11::module xtensor xtensor::optimize xtensor-python xtensor::use_xsimd Python::NumPy tbb)

target_compile_definitions(mymodule PRIVATE VERSION_INFO=0.1.0)

# copy run script in build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/run.sh DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/)