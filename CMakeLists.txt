cmake_minimum_required(VERSION 3.15)
project(HopfNetMHD)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyBind11, Eigen3, and Python
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Create the Python module
pybind11_add_module(hopfnet_cpp src/bindings.cpp)

# Link Eigen3
target_link_libraries(hopfnet_cpp PRIVATE Eigen3::Eigen)