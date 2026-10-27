#pragma once

constexpr int silhouette_kernel_count = 6;
int silhouette_benchmark_configuration_count();

void silhouette_samples_bench(const int* labels, const float* distances, float* scores,
                              int sample_count, int cluster_count, float* kernel_times_ms);

const char* silhouette_benchmark_configuration_name(int configuration_index);

void silhouette_samples_bench_configuration(int configuration_index, const int* labels,
                                            const float* distances, float* scores,
                                            int sample_count, int cluster_count,
                                            float* kernel_times_ms);
