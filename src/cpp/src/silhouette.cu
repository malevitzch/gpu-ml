#include "silhouette.cuh"

void silhouette_samples(const int* labels, const float* distances, float* scores,
                        std::size_t sample_count, std::size_t cluster_count) {
    for (std::size_t i = 0; i < sample_count; ++i) {
        scores[i] = 0.0f;
    }
}
