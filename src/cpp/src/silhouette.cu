#include "silhouette.cuh"

void silhouette_samples(const int* labels, const float* distances, float* scores,
                        std::size_t sample_count, std::size_t cluster_count) {
    for (std::size_t i = 0; i < sample_count; ++i) {
        scores[i] = 0.0f;
    }
}

// This version assumes that there are relatively many clusters
// Shouldn't be a bottleneck though as it's just O(n) computation as opposed to
// the O(n^2) computations that are done later, so it's kept simple for now
__global__ void cluster_sizes_atomic_kernel(const int* __restrict__ labels,
                              int* __restrict__ sizes, int sample_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < sample_count) atomicAdd(&sizes[labels[idx]], 1);
}


// This is a largely inefficient version, a manual reduction over each row
// would probably be faster
__global__ void a_array_atomic_kernel(const int* __restrict__ labels,
                               float* __restrict__ distances, int* __restrict__ cluster_sizes,
                               float* __restrict__ a, int sample_count) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < sample_count && y < sample_count) {
        int contrib = (labels[x] == labels[y] && x != y);
        atomicAdd(&a[x], contrib * distances[sample_count * y + x]);
    }
}
