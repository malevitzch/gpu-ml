#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>
#include "addkernel.cuh"

namespace py = pybind11;

void add_scalar(py::array_t<float> arr, float scalar) {
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);

    // Allocate on device, run kernel, copy back
    float* d_data;
    size_t bytes = buf.size * sizeof(float);
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, ptr, bytes, cudaMemcpyHostToDevice);
    launch_add_scalar(d_data, scalar, buf.size);
    cudaMemcpy(ptr, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

PYBIND11_MODULE(_core, m) {
    m.def("add_scalar", &add_scalar, "Add a scalar to a float32 numpy array in-place");
}