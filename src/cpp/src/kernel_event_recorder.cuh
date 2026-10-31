#pragma once

#include <cuda_runtime.h>

#include "silhouette_bench.cuh"

class KernelEventRecorder {
public:
    int start() {
        int index = count_++;
        cudaEventCreate(&starts_[index]);
        cudaEventCreate(&stops_[index]);
        cudaEventRecord(starts_[index]);
        return index;
    }

    void stop(int index) {
        cudaEventRecord(stops_[index]);
    }

    void collect(float* elapsed_ms) {
        cudaDeviceSynchronize();
        for (int i = 0; i < count_; ++i) {
            cudaEventElapsedTime(&elapsed_ms[i], starts_[i], stops_[i]);
            cudaEventDestroy(starts_[i]);
            cudaEventDestroy(stops_[i]);
        }
    }

private:
    cudaEvent_t starts_[silhouette_kernel_count];
    cudaEvent_t stops_[silhouette_kernel_count];
    int count_ = 0;
};
