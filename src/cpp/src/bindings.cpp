#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "addkernel.cuh"

namespace py = pybind11;

void add_scalar_binding(py::array_t<float> arr, float scalar) {
    auto buf = arr.request();
    add_scalar(static_cast<float*>(buf.ptr), scalar, static_cast<int>(buf.size));
}

PYBIND11_MODULE(_core, m) {
    m.def("add_scalar", &add_scalar_binding, "Add a scalar to a float32 numpy array in-place");
}
