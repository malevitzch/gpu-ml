#include <cstddef>

void silhouette_samples(const int* labels, const float* distances, float* scores,
                        std::size_t sample_count, std::size_t cluster_count);
