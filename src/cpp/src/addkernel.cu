#include <cuda_runtime.h>

__global__ void add_scalar_kernel(float* data, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += scalar;
}

static void launch_add_scalar_kernel(float* data, float scalar, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_scalar_kernel<<<blocks, threads>>>(data, scalar, n);
    cudaDeviceSynchronize();
}

void add_scalar(float* data, float scalar, int n) {
    float* d_data;
    size_t bytes = n * sizeof(float);
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);
    launch_add_scalar_kernel(d_data, scalar, n);
    cudaMemcpy(data, d_data, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
