#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "silhouette.cuh"

namespace py = pybind11;
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
                       static_cast<int>(labelbuf.size), cluster_count);
    return scores;
}

PYBIND11_MODULE(_core, m) {
    m.def("silhouette_samples", &silhouette_samples_binding,
          py::arg("labels"), py::arg("distances"), py::arg("cluster_count"),
          "Compute one silhouette coefficient per sample");
}
