#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "addkernel.cuh"
#include "silhouette.cuh"

namespace py = pybind11;

void add_scalar_binding(py::array_t<float> arr, float scalar) {
    auto buf = arr.request();
    add_scalar(static_cast<float*>(buf.ptr), scalar, static_cast<int>(buf.size));
}

py::array_t<float> silhouette_samples_binding(py::array_t<int> labels,
                                                py::array_t<float> distances,
                                                int cluster_count) {
    auto labelbuf = labels.request();
    auto distbuf = distances.request();
    py::array_t<float> scores(labelbuf.size);
    auto scorebuf = scores.request();

    silhouette_samples(static_cast<const int*>(labelbuf.ptr),
                       static_cast<const float*>(distbuf.ptr),
                       static_cast<float*>(scorebuf.ptr),
                       static_cast<std::size_t>(labelbuf.size),
                       static_cast<std::size_t>(cluster_count));
    return scores;
}

PYBIND11_MODULE(_core, m) {
    m.def("add_scalar", &add_scalar_binding, "Add a scalar to a float32 numpy array in-place");
    m.def("silhouette_samples", &silhouette_samples_binding,
          py::arg("labels"), py::arg("distances"), py::arg("cluster_count"),
          "Compute one silhouette coefficient per sample");
}
