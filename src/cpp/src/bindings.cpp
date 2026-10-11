#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "addkernel.cuh"
#include "silhouette.cuh"

namespace py = pybind11;

void add_scalar_binding(py::array_t<float> arr, float scalar) {
    auto buf = arr.request();
    add_scalar(static_cast<float*>(buf.ptr), scalar, static_cast<int>(buf.size));
}

float silhouette_binding(py::array_t<int> labels, int label_count, py::array_t<float> distances) {

    auto labelbuf = labels.request();
    int* labels_ptr = static_cast<int*>(labelbuf.ptr);
    auto distbuf = distances.request();
    float* distances_ptr = static_cast<float*>(distbuf.ptr);

    return silhouette(labels_ptr, distances_ptr, labelbuf.size, label_count);
}

PYBIND11_MODULE(_core, m) {
    m.def("add_scalar", &add_scalar_binding, "Add a scalar to a float32 numpy array in-place");
    m.def("silhouette", &silhouette_binding,
          "Compute the average silhouette score from labels and pairwise distances");
}
