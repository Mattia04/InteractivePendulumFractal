cmake_minimum_required(VERSION 3.30)
project(InteractivePendulumFractal)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -flto")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

# Find required packages
find_package(Boost REQUIRED)
find_package(OpenCL REQUIRED)
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCL_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})

# Add the pybind11 module
pybind11_add_module(simulation sources/simulation.cpp)

# Link libraries
target_link_libraries(simulation
    PRIVATE
    ${OpenCL_LIBRARIES}
)

# Copy the OpenCL kernel file to the build directory
configure_file(
    ${CMAKE_SOURCE_DIR}/sources/kernel.cl
    ${CMAKE_BINARY_DIR}/kernel.cl
    COPYONLY
)