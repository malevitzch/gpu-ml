// Compile the production implementation into this translation unit so the
// benchmark-only launch wrappers below use its exact kernel definitions.
#include "silhouette.cu"
#include "silhouette_bench.cuh"

#include <tuple>

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

// Each launch policy is intentionally a type rather than a runtime function
// pointer: a benchmark configuration is fully selected at compile time.
struct ClusterSizesAtomicLaunch {
    static void launch(const int* labels, int* sizes, int sample_count,
                       KernelEventRecorder& events) {
        constexpr int block_size = 1024;
        const int event = events.start();
        cluster_sizes_atomic_kernel<<<(sample_count + block_size - 1) / block_size,
                                      block_size>>>(labels, sizes, sample_count);
        events.stop(event);
    }
};

template <int TileSize>
struct AArrayAtomicLaunch {
    static void launch(const int* labels, const float* distances, float* a, int sample_count,
                       KernelEventRecorder& events) {
        const dim3 block(TileSize, TileSize);
        const dim3 grid((sample_count + TileSize - 1) / TileSize,
                        (sample_count + TileSize - 1) / TileSize);
        const int event = events.start();
        a_array_atomic_kernel<<<grid, block>>>(labels, distances, a, sample_count);
        events.stop(event);
    }
};

struct NormalizeALaunch {
    static void launch(float* a, const int* cluster_sizes, const int* labels,
                       int sample_count, KernelEventRecorder& events) {
        constexpr int block_size = 1024;
        const int event = events.start();
        normalize_a_kernel<<<(sample_count + block_size - 1) / block_size, block_size>>>(
            a, cluster_sizes, labels, sample_count);
        events.stop(event);
    }
};

struct ClusterDistancesAtomicLaunch {
    static void launch(const int* labels, const float* distances, float* cluster_results,
                       int sample_count, int cluster_count, KernelEventRecorder& events) {
        const dim3 block(32, 32);
        const dim3 grid((sample_count + block.x - 1) / block.x,
                        (sample_count + block.y - 1) / block.y);
        const int event = events.start();
        cluster_distances_kernel<<<grid, block>>>(
            labels, distances, cluster_results, sample_count, cluster_count);
        events.stop(event);
    }
};

struct BArrayLaunch {
    static void launch(const int* labels, const float* cluster_results,
                       const int* cluster_sizes, float* b, int sample_count,
                       int cluster_count, KernelEventRecorder& events) {
        constexpr int block_size = 1024;
        const int event = events.start();
        b_kernel<<<(sample_count + block_size - 1) / block_size, block_size>>>(
            labels, cluster_results, cluster_sizes, b, sample_count, cluster_count);
        events.stop(event);
    }
};

struct SilhouetteLaunch {
    static void launch(float* a, const float* b, float* scores, int sample_count,
                       KernelEventRecorder& events) {
        constexpr int block_size = 1024;
        const int event = events.start();
        silhouette_kernel<<<(sample_count + block_size - 1) / block_size, block_size>>>(
            a, b, scores, sample_count);
        events.stop(event);
    }
};

// Experimental kernels and their launch policies live separately from production.
#include "silhouette_experiments.cuh"

template <typename ClusterSizesLaunch, typename AArrayLaunch, typename NormalizeLaunch,
          typename ClusterDistancesLaunch, typename BLaunch, typename ScoreLaunch>
struct SilhouetteBenchmarkConfiguration {
    static void run(const int* labels, const float* distances, float* scores,
                    int sample_count, int cluster_count, float* kernel_times_ms) {
        int* device_labels;
        float* device_distances;
        cudaMalloc(&device_labels, sizeof(int) * sample_count);
        cudaMalloc(&device_distances, sizeof(float) * sample_count * sample_count);
        cudaMemcpy(device_labels, labels, sizeof(int) * sample_count, cudaMemcpyHostToDevice);
        cudaMemcpy(device_distances, distances, sizeof(float) * sample_count * sample_count,
                   cudaMemcpyHostToDevice);

        KernelEventRecorder events;
        int* cluster_sizes;
        cudaMalloc(&cluster_sizes, cluster_count * sizeof(int));
        cudaMemset(cluster_sizes, 0, cluster_count * sizeof(int));
        ClusterSizesLaunch::launch(device_labels, cluster_sizes, sample_count, events);

        float* a;
        cudaMalloc(&a, sizeof(float) * sample_count);
        cudaMemset(a, 0, sizeof(float) * sample_count);
        AArrayLaunch::launch(device_labels, device_distances, a, sample_count, events);
        NormalizeLaunch::launch(a, cluster_sizes, device_labels, sample_count, events);

        float* cluster_results;
        cudaMalloc(&cluster_results, sizeof(float) * sample_count * cluster_count);
        cudaMemset(cluster_results, 0, sizeof(float) * sample_count * cluster_count);
        float* b;
        cudaMalloc(&b, sizeof(float) * sample_count);
        ClusterDistancesLaunch::launch(device_labels, device_distances, cluster_results,
                                       sample_count, cluster_count, events);
        BLaunch::launch(device_labels, cluster_results, cluster_sizes, b, sample_count,
                        cluster_count, events);

        float* device_scores;
        cudaMalloc(&device_scores, sizeof(float) * sample_count);
        ScoreLaunch::launch(a, b, device_scores, sample_count, events);
        cudaMemcpy(scores, device_scores, sizeof(float) * sample_count, cudaMemcpyDeviceToHost);
        events.collect(kernel_times_ms);

        cudaFree(device_scores);
        cudaFree(b);
        cudaFree(cluster_results);
        cudaFree(a);
        cudaFree(cluster_sizes);
        cudaFree(device_distances);
        cudaFree(device_labels);
    }
};

struct BaselineConfiguration : SilhouetteBenchmarkConfiguration<
    ClusterSizesAtomicLaunch, AArrayAtomicLaunch<32>, NormalizeALaunch,
    ClusterDistancesAtomicLaunch, BArrayLaunch, SilhouetteLaunch> {
    static constexpr const char* name = "baseline_a_tile_32";
};

struct AArrayTile16Configuration : SilhouetteBenchmarkConfiguration<
    ClusterSizesAtomicLaunch, AArrayAtomicLaunch<16>, NormalizeALaunch,
    ClusterDistancesAtomicLaunch, BArrayLaunch, SilhouetteLaunch> {
    static constexpr const char* name = "a_tile_16";
};

struct AArrayRowConfiguration : SilhouetteBenchmarkConfiguration<
    ClusterSizesAtomicLaunch, AArrayRowLaunch, NormalizeALaunch,
    ClusterDistancesAtomicLaunch, BArrayLaunch, SilhouetteLaunch> {
    static constexpr const char* name = "a_row";
};

// Add future fully-instantiated launch configurations here. This is the single
// compile-time registry consumed by the Python configuration benchmark.
using BenchmarkConfigurations = std::tuple<BaselineConfiguration, AArrayTile16Configuration,
                                           AArrayRowConfiguration>;

template <std::size_t Index = 0>
const char* configuration_name(int configuration_index) {
    if (configuration_index == static_cast<int>(Index)) {
        return std::tuple_element_t<Index, BenchmarkConfigurations>::name;
    }
    if constexpr (Index + 1 < std::tuple_size<BenchmarkConfigurations>::value) {
        return configuration_name<Index + 1>(configuration_index);
    }
    return nullptr;
}

template <std::size_t Index = 0>
void run_configuration(int configuration_index, const int* labels, const float* distances,
                       float* scores, int sample_count, int cluster_count,
                       float* kernel_times_ms) {
    if (configuration_index == static_cast<int>(Index)) {
        std::tuple_element_t<Index, BenchmarkConfigurations>::run(
            labels, distances, scores, sample_count, cluster_count, kernel_times_ms);
        return;
    }
    if constexpr (Index + 1 < std::tuple_size<BenchmarkConfigurations>::value) {
        run_configuration<Index + 1>(configuration_index, labels, distances, scores,
                                     sample_count, cluster_count, kernel_times_ms);
    }
}

}  // namespace

void silhouette_samples_bench(const int* labels, const float* distances, float* scores,
                              int sample_count, int cluster_count, float* kernel_times_ms) {
    BaselineConfiguration::run(labels, distances, scores, sample_count, cluster_count,
                               kernel_times_ms);
}

const char* silhouette_benchmark_configuration_name(int configuration_index) {
    return configuration_name(configuration_index);
}

int silhouette_benchmark_configuration_count() {
    return static_cast<int>(std::tuple_size<BenchmarkConfigurations>::value);
}

void silhouette_samples_bench_configuration(int configuration_index, const int* labels,
                                            const float* distances, float* scores,
                                            int sample_count, int cluster_count,
                                            float* kernel_times_ms) {
    run_configuration(configuration_index, labels, distances, scores, sample_count,
                      cluster_count, kernel_times_ms);
}
