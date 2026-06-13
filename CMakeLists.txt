cmake_minimum_required(VERSION 3.15)
project(HopfNetMHD)

# Set C++20 standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CRUCIAL Apple Silicon Fix: Append the Homebrew path to CMake's search list
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
endif()

# Find PyBind11, Eigen3, and Python components
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find FFTW3 (installed via `brew install fftw`)
find_path(FFTW3_INCLUDE_DIR fftw3.h PATHS /opt/homebrew/include)
find_library(FFTW3_LIB fftw3 PATHS /opt/homebrew/lib)

if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIB)
    message(FATAL_ERROR "FFTW3 not found. Install with: brew install fftw")
endif()

# Create the Python module
pybind11_add_module(hopfnet_cpp src/bindings.cpp)

target_include_directories(hopfnet_cpp PRIVATE ${FFTW3_INCLUDE_DIR} src)

# Link Eigen3 and FFTW3
target_link_libraries(hopfnet_cpp PRIVATE Eigen3::Eigen ${FFTW3_LIB})
