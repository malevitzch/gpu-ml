#include <cuda_runtime.h>

__global__ void add_scalar_kernel(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += scalar;
}

void launch_add_scalar(float* data, float scalar, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads>>>(data, scalar, n);
    cudaDeviceSynchronize();
}