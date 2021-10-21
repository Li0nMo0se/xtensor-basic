cmake_minimum_required(VERSION 3.18..3.20)

project(mymodule)

set(XTENSOR_USE_XSIMD 1)
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 REQUIRED CONFIG)
find_package(xtensor REQUIRED)
find_package(xtensor-python REQUIRED)


# /!\ Notice that this option prevents building on a machine and distributing
# the resulting binary on another machine with a different architecture
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -mavx2 -ffast-math")

pybind11_add_module(mymodule main.cpp)
target_link_libraries(mymodule PUBLIC pybind11::module xtensor xtensor-python Python::NumPy)

target_compile_definitions(mymodule PRIVATE VERSION_INFO=0.1.0)