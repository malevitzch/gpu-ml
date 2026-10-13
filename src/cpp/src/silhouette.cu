#include <cuda_runtime.h>
#include <math_constants.h>
#include "silhouette.cuh"


// This version assumes that there are relatively many clusters
// Shouldn't be a bottleneck though as it's just O(n) computation as opposed to
// the O(n^2) computations that are done later, so it's kept simple for now
__global__ void cluster_sizes_atomic_kernel(const int* __restrict__ labels,
                              int* __restrict__ sizes, int sample_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < sample_count) atomicAdd(&sizes[labels[idx]], 1);
}

int* get_cluster_sizes(const int* labels, int sample_count, int cluster_count) {
    int* sizes;
    cudaMalloc(&sizes, cluster_count * sizeof(int));
    cudaMemset(sizes, 0, cluster_count * sizeof(int));
    int block_size = 1024;
    int grid_size = (sample_count + block_size - 1) / block_size;
    cluster_sizes_atomic_kernel<<<grid_size, block_size>>>(labels, sizes, sample_count);
    return sizes;
}


// This is a largely inefficient version, a manual reduction over each row
// would probably be faster
__global__ void a_array_atomic_kernel(const int* __restrict__ labels,
                               const float* __restrict__ distances,
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

float* compute_a_array(const int* labels, const float* distances, const int* cluster_sizes, int sample_count) {
    float* a;
    cudaMalloc(&a, sizeof(int) * sample_count);
    cudaMemset(a, 0, sizeof(int) * sample_count);
    dim3 grid_dim((sample_count + 31) / 32, (sample_count + 31) / 32);
    dim3 block_dim = {32, 32};
    a_array_atomic_kernel<<<grid_dim, block_dim>>>(labels, distances, a, sample_count);
    normalize_a_kernel<<<(sample_count + 1023) / 1024, 1024>>>(a, cluster_sizes, labels, sample_count);
    return a;
}

// b computation
// 1. compute distances from each sample to each cluster.
// Options:
//  - a) generate samples x clusters array of distances (not a lot of clusters)
//  - b) Kernel-per-sample
//  - c) Kernel-per-cluster (This actually doesn't work too well)
//  - single kernel per distance (?)
// 2. compute min across sample, cluster


// Assumes cluster_results is zeroed out before calling
__global__ void cluster_distances_kernel(const int* __restrict__ labels,
                                         const float* __restrict__ distances,
                                         float* __restrict__ result,
                                         int sample_count,
                                         int cluster_count) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < sample_count && y < sample_count) {
        if(labels[x] != labels[y]) {
            atomicAdd(&result[x * cluster_count + labels[y]], distances[sample_count * y + x]);
        }
    }
}

// TODO: move cluster_sizes to shared memory? This should avoid unnecessary loads
__global__ void b_kernel(const int* __restrict__  labels,
                         const float* __restrict__ cdists,
                         const int* __restrict__ cluster_sizes,
                         float* __restrict__ b,
                         int sample_count, int cluster_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < sample_count) {
        int label = labels[idx];
        float result = CUDART_INF_F;
        for(int i = 0; i < cluster_count; i++) {
            if(i != label) {
                result = min(result, cdists[idx * cluster_count + i] / cluster_sizes[i]);
            }
        }
        b[idx] = result;
    }
}

float* compute_b_array(const int* labels, const float* distances,
                       const int* cluster_sizes, int sample_count, int cluster_count) {
    dim3 block_size = {32, 32};
    dim3 grid_dim = {(sample_count + block_size.x - 1) / block_size.x, (sample_count + block_size.y - 1) / block_size.y};
    float* cluster_results;
    cudaMalloc(&cluster_results, sizeof(float) * sample_count * cluster_count);
    cudaMemset(cluster_results, 0, sizeof(float) * sample_count * cluster_count);
    float* b;
    cudaMalloc(&b, sizeof(float) * sample_count);
    cluster_distances_kernel<<<grid_dim, block_size>>>(labels, distances, cluster_results, sample_count, cluster_count);
    b_kernel<<<(sample_count + 1023) / 1024, 1024>>>(labels, cluster_results, cluster_sizes, b, sample_count, cluster_count);
    return cluster_results;
}


__global__ void silhouette_kernel(float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ scores, int sample_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < sample_count) {
        float ai = a[idx];
        float bi = b[idx];
        scores[idx] = (bi - ai) / max(ai, bi);
    }
}

void silhouette_samples(const int* labels, const float* distances, float* scores,
                        int sample_count, int cluster_count) {
    int* cluster_sizes = get_cluster_sizes(labels, sample_count, cluster_count);
    float* a = compute_a_array(labels, distances, cluster_sizes, sample_count);
    float* b = compute_b_array(labels, distances, cluster_sizes, sample_count, cluster_count);
    float* device_scores;
    cudaMalloc(&device_scores, sizeof(float) * sample_count);
    silhouette_kernel<<<(sample_count + 1023) / 1024, 1024>>>(a, b, device_scores, sample_count);
    cudaMemcpy(scores, device_scores, sizeof(float) * sample_count, cudaMemcpyDeviceToHost);
    cudaFree(device_scores);
    cudaFree(a);
    cudaFree(b);
    cudaFree(cluster_sizes);
}
