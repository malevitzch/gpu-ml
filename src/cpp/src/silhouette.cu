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
                               float* __restrict__ distances,
                               float* __restrict__ a, int sample_count) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < sample_count && y < sample_count) {
        int contrib = (labels[x] == labels[y] && x != y);
        atomicAdd(&a[x], contrib * distances[sample_count * y + x]);
    }
}

// This is a small kernel that takes no time since it's O(n) and we do O(n^2) things
__global__ void normalize_a_kernel(float* __restrict__ a,
                                   const int* __restrict__ sizes, const int* __restrict__ labels,
                                   int sample_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < sample_count && sizes[labels[idx]] > 1) {
        a[idx] /= (sizes[labels[idx]] - 1);
    }
}

// b computation
// 1. compute distances from each sample to each cluster.
// Options:
//  - Kernel-per-sample
//  - Kernel-per-cluster (This actually doesn't work too well)
//  - single kernel per distance (?)
// 2. compute min across sample, cluster


// Those are still not normalized and min is not applied. It will be done in a secondary kernel
__global__ void b_i_cluster_distances_kernel(const int* __restrict__ labels,
                           const float* __restrict__ distances,
                           float* __restrict__ cluster_results,
                           int i,
                           int sample_count,
                           int cluster_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cluster = labels[i];
    if(idx < sample_count) {
        if(labels[idx] != cluster) {
            atomicAdd(&cluster_results[labels[idx]], distances[i * sample_count + idx]);
        }
    }
}


// Performance is not too critical there as it's O(C), and there are not too many clusters
// in any reasonable use case.
__global__ void normalize_and_agg_kernel(float* __restrict__ cluster_results,
                                         const int* __restrict__ cluster_sizes, int cluster_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < cluster_count) {
        cluster_results[idx] /= cluster_sizes[idx];
        // TODO: reduction
    }
}