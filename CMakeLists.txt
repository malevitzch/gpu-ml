cmake_minimum_required(VERSION 3.24)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compile commands for clangd")

# Use the same set of GPU targets on every build host. Override this cache
# variable when targeting a smaller or different set of GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90"
    CACHE STRING "CUDA architectures to compile for")
endif()

project(frogml LANGUAGES CXX CUDA)

find_package(CUDAToolkit 12.0 REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core NO_EXTRAS
    src/cpp/src/bindings.cpp
    src/cpp/src/silhouette_bench.cu
)

target_include_directories(_core PRIVATE src/cpp/include)

target_compile_features(_core PRIVATE cxx_std_17)

install(TARGETS _core DESTINATION frogml)
