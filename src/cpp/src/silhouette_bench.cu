// Compile the production implementation into this translation unit so the
// benchmark-only launch wrappers below use its exact kernel definitions.
#include "silhouette.cu"
#include "silhouette_bench.cuh"

namespace {

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

int* get_cluster_sizes_bench(const int* labels, int sample_count, int cluster_count,
                             KernelEventRecorder& events) {
    int* sizes;
    cudaMalloc(&sizes, cluster_count * sizeof(int));
    cudaMemset(sizes, 0, cluster_count * sizeof(int));
    int block_size = 1024;
    int grid_size = (sample_count + block_size - 1) / block_size;
    int event = events.start();
    cluster_sizes_atomic_kernel<<<grid_size, block_size>>>(labels, sizes, sample_count);
    events.stop(event);
    return sizes;
}

float* compute_a_array_bench(const int* labels, const float* distances,
                             const int* cluster_sizes, int sample_count,
                             KernelEventRecorder& events) {
    float* a;
    cudaMalloc(&a, sizeof(float) * sample_count);
    cudaMemset(a, 0, sizeof(float) * sample_count);
    dim3 grid_dim((sample_count + 31) / 32, (sample_count + 31) / 32);
    dim3 block_dim(32, 32);
    int event = events.start();
    a_array_atomic_kernel<<<grid_dim, block_dim>>>(labels, distances, a, sample_count);
    events.stop(event);
    event = events.start();
    normalize_a_kernel<<<(sample_count + 1023) / 1024, 1024>>>(
        a, cluster_sizes, labels, sample_count);
    events.stop(event);
    return a;
}

float* compute_b_array_bench(const int* labels, const float* distances,
                             const int* cluster_sizes, int sample_count,
                             int cluster_count, KernelEventRecorder& events) {
    dim3 block_size(32, 32);
    dim3 grid_dim((sample_count + block_size.x - 1) / block_size.x,
                  (sample_count + block_size.y - 1) / block_size.y);
    float* cluster_results;
    cudaMalloc(&cluster_results, sizeof(float) * sample_count * cluster_count);
    cudaMemset(cluster_results, 0, sizeof(float) * sample_count * cluster_count);
    float* b;
    cudaMalloc(&b, sizeof(float) * sample_count);
    int event = events.start();
    cluster_distances_kernel<<<grid_dim, block_size>>>(
        labels, distances, cluster_results, sample_count, cluster_count);
    events.stop(event);
    event = events.start();
    b_kernel<<<(sample_count + 1023) / 1024, 1024>>>(
        labels, cluster_results, cluster_sizes, b, sample_count, cluster_count);
    events.stop(event);
    return b;
}

}  // namespace

void silhouette_samples_bench(const int* labels, const float* distances, float* scores,
                              int sample_count, int cluster_count, float* kernel_times_ms) {
    KernelEventRecorder events;
    int* device_labels;
    float* device_distances;
    cudaMalloc(&device_labels, sizeof(int) * sample_count);
    cudaMalloc(&device_distances, sizeof(float) * sample_count * sample_count);
    cudaMemcpy(device_labels, labels, sizeof(int) * sample_count, cudaMemcpyHostToDevice);
    cudaMemcpy(device_distances, distances, sizeof(float) * sample_count * sample_count,
               cudaMemcpyHostToDevice);

    int* cluster_sizes = get_cluster_sizes_bench(device_labels, sample_count, cluster_count,
                                                  events);
    float* a = compute_a_array_bench(device_labels, device_distances, cluster_sizes,
                                     sample_count, events);
    float* b = compute_b_array_bench(device_labels, device_distances, cluster_sizes,
                                     sample_count, cluster_count, events);
    float* device_scores;
    cudaMalloc(&device_scores, sizeof(float) * sample_count);
    int event = events.start();
    silhouette_kernel<<<(sample_count + 1023) / 1024, 1024>>>(a, b, device_scores,
                                                             sample_count);
    events.stop(event);
    cudaMemcpy(scores, device_scores, sizeof(float) * sample_count, cudaMemcpyDeviceToHost);
    events.collect(kernel_times_ms);

    cudaFree(device_scores);
    cudaFree(a);
    cudaFree(b);
    cudaFree(cluster_sizes);
    cudaFree(device_distances);
    cudaFree(device_labels);
}
