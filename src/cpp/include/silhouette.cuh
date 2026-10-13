#pragma once

void silhouette_samples(const int* labels, const float* distances, float* scores,
                        int sample_count, int cluster_count);
