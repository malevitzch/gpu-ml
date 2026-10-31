#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

#include "kernel_event_recorder.cuh"

// Independent copies of the production kernels and their launch policies.
// Edit these to compare a complete implementation against the baseline.
namespace {

__global__ void editable_cluster_sizes_kernel(const int* __restrict__ labels,
                                               int* __restrict__ sizes, int sample_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < sample_count) atomicAdd(&sizes[labels[idx]], 1);
}

__global__ void editable_a_array_kernel(const int* __restrict__ labels,
                                        const float* __restrict__ distances,
                                        float* __restrict__ a, int sample_count) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < sample_count && y < sample_count) {
        int contrib = (labels[x] == labels[y] && x != y);
        atomicAdd(&a[x], contrib * distances[sample_count * y + x]);
    }
}

__global__ void editable_normalize_a_kernel(float* __restrict__ a,
                                             const int* __restrict__ sizes,
                                             const int* __restrict__ labels,
                                             int sample_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < sample_count && sizes[labels[idx]] > 1) {
        a[idx] /= (sizes[labels[idx]] - 1);
    }
}

__global__ void editable_cluster_distances_kernel(const int* __restrict__ labels,
                                                   const float* __restrict__ distances,
                                                   float* __restrict__ result,
                                                   int sample_count,
                                                   int cluster_count) {
    __shared__ int labels_x[32];
    __shared__ int labels_y[32];
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(threadIdx.y == 0) {
        int index = blockIdx.y * blockDim.y + threadIdx.x;
        labels_y[threadIdx.x] = index < sample_count ? labels[index] : 0;
    }
    if(threadIdx.y == 31) {
        labels_x[threadIdx.x] = x < sample_count ? labels[x] : 0;
    }

    __syncthreads();

    if(x < sample_count && y < sample_count) {
        if(labels_x[threadIdx.x] != labels_y[threadIdx.y]) {
            atomicAdd(&result[x * cluster_count + labels_y[threadIdx.y]], distances[sample_count * y + x]);
        }
    }
}

__global__ void editable_b_kernel(const int* __restrict__ labels,
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

__global__ void editable_silhouette_kernel(float* __restrict__ a, const float* __restrict__ b,
                                            float* __restrict__ scores, int sample_count) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < sample_count) {
        float ai = a[idx];
        float bi = b[idx];
        if(max(ai, bi) == 0.0f) {
            scores[idx] = 0.0f;
        } else {
            scores[idx] = (bi - ai) / max(ai, bi);
        }
    }
}

struct EditableClusterSizesLaunch {
    static void launch(const int* labels, int* sizes, int sample_count,
                       KernelEventRecorder& events) {
        constexpr int block_size = 1024;
        const int event = events.start();
        editable_cluster_sizes_kernel<<<(sample_count + block_size - 1) / block_size,
                                        block_size>>>(labels, sizes, sample_count);
        events.stop(event);
    }
};

struct EditableAArrayLaunch {
    static void launch(const int* labels, const float* distances, float* a, int sample_count,
                       KernelEventRecorder& events) {
        const dim3 block(32, 32);
        const dim3 grid((sample_count + block.x - 1) / block.x,
                        (sample_count + block.y - 1) / block.y);
        const int event = events.start();
        editable_a_array_kernel<<<grid, block>>>(labels, distances, a, sample_count);
        events.stop(event);
    }
};

struct EditableNormalizeALaunch {
    static void launch(float* a, const int* cluster_sizes, const int* labels,
                       int sample_count, KernelEventRecorder& events) {
        constexpr int block_size = 1024;
        const int event = events.start();
        editable_normalize_a_kernel<<<(sample_count + block_size - 1) / block_size, block_size>>>(
            a, cluster_sizes, labels, sample_count);
        events.stop(event);
    }
};

struct EditableClusterDistancesLaunch {
    static void launch(const int* labels, const float* distances, float* cluster_results,
                       int sample_count, int cluster_count, KernelEventRecorder& events) {
        const dim3 block(32, 32);
        const dim3 grid((sample_count + block.x - 1) / block.x,
                        (sample_count + block.y - 1) / block.y);
        const int event = events.start();
        editable_cluster_distances_kernel<<<grid, block>>>(
            labels, distances, cluster_results, sample_count, cluster_count);
        events.stop(event);
    }
};

struct EditableBArrayLaunch {
    static void launch(const int* labels, const float* cluster_results,
                       const int* cluster_sizes, float* b, int sample_count,
                       int cluster_count, KernelEventRecorder& events) {
        constexpr int block_size = 1024;
        const int event = events.start();
        editable_b_kernel<<<(sample_count + block_size - 1) / block_size, block_size>>>(
            labels, cluster_results, cluster_sizes, b, sample_count, cluster_count);
        events.stop(event);
    }
};

struct EditableSilhouetteLaunch {
    static void launch(float* a, const float* b, float* scores, int sample_count,
                       KernelEventRecorder& events) {
        constexpr int block_size = 1024;
        const int event = events.start();
        editable_silhouette_kernel<<<(sample_count + block_size - 1) / block_size, block_size>>>(
            a, b, scores, sample_count);
        events.stop(event);
    }
};

}  // namespace
