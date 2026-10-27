// Benchmark-only kernel variants. This file is included after the launch
// policies and KernelEventRecorder in silhouette_bench.cu.

// One thread accumulates a sample's distances, avoiding atomic additions.
// Like the baseline, this produces an unnormalized sum for NormalizeALaunch.
__global__ void a_array_row_kernel(const int* labels, const float* distances,
                                   float* a, int sample_count) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= sample_count) return;
    float sum = 0.0f;
    for (int y = 0; y < sample_count; ++y) {
        if (x != y && labels[x] == labels[y]) {
            sum += distances[static_cast<size_t>(y) * sample_count + x];
        }
    }
    a[x] = sum;
}

struct AArrayRowLaunch {
    static void launch(const int* labels, const float* distances, float* a,
                       int sample_count, KernelEventRecorder& events) {
        constexpr int block_size = 256;
        const int event = events.start();
        a_array_row_kernel<<<(sample_count + block_size - 1) / block_size, block_size>>>(
            labels, distances, a, sample_count);
        events.stop(event);
    }
};
