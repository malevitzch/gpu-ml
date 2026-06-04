cmake_minimum_required(VERSION 3.18)
project(your_package LANGUAGES CXX CUDA)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module NumPy)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core
    src/cpp/src/bindings.cpp
    src/cpp/src/addkernel.cu
)

set_target_properties(_core PROPERTIES
    INTERPROCEDURAL_OPTIMIZATION OFF
)
target_include_directories(_core PRIVATE src/cpp/include)

target_compile_features(_core PRIVATE cxx_std_17)
set_target_properties(_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(_core PRIVATE ${Python_NumPy_INCLUDE_DIRS})

install(TARGETS _core DESTINATION your_package)