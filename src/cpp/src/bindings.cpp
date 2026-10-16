#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "silhouette.cuh"
#include "silhouette_bench.cuh"

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

py::tuple silhouette_samples_bench_binding(py::array_t<int> labels,
                                           py::array_t<float> distances,
                                           int cluster_count) {
    auto labelbuf = labels.request();
    auto distbuf = distances.request();
    py::array_t<float> scores(labelbuf.size);
    py::array_t<float> kernel_times(silhouette_kernel_count);
    auto scorebuf = scores.request();
    auto timingsbuf = kernel_times.request();

    silhouette_samples_bench(static_cast<const int*>(labelbuf.ptr),
                             static_cast<const float*>(distbuf.ptr),
                             static_cast<float*>(scorebuf.ptr),
                             static_cast<int>(labelbuf.size), cluster_count,
                             static_cast<float*>(timingsbuf.ptr));

    const char* kernel_names[] = {
        "cluster_sizes_atomic_kernel",
        "a_array_atomic_kernel",
        "normalize_a_kernel",
        "cluster_distances_kernel",
        "b_kernel",
        "silhouette_kernel",
    };
    py::dict timings;
    const float* timing_values = static_cast<const float*>(timingsbuf.ptr);
    for (int i = 0; i < silhouette_kernel_count; ++i) {
        timings[py::str(kernel_names[i])] = timing_values[i];
    }
    return py::make_tuple(scores, timings);
}

PYBIND11_MODULE(_core, m) {
    m.def("silhouette_samples", &silhouette_samples_binding,
          py::arg("labels"), py::arg("distances"), py::arg("cluster_count"),
          "Compute one silhouette coefficient per sample");
    m.def("silhouette_samples_bench", &silhouette_samples_bench_binding,
          py::arg("labels"), py::arg("distances"), py::arg("cluster_count"),
          "Return scores and per-kernel CUDA event timings in milliseconds");
}
