#pragma once

inline constexpr int silhouette_kernel_count = 6;

void silhouette_samples_bench(const int* labels, const float* distances, float* scores,
                              int sample_count, int cluster_count, float* kernel_times_ms);
