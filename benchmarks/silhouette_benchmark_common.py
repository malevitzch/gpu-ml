"""Shared data generation and reporting for silhouette CUDA benchmarks."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import pairwise_distances


def make_data(sample_count: int, cluster_count: int, feature_count: int, seed: int):
    rng = np.random.default_rng(seed)
    labels = np.arange(sample_count, dtype=np.int32) % cluster_count
    rng.shuffle(labels)
    centers = rng.normal(0.0, 8.0, size=(cluster_count, feature_count)).astype(np.float32)
    features = centers[labels] + rng.normal(0.0, 1.0, size=(sample_count, feature_count)).astype(np.float32)
    distances = pairwise_distances(features, metric="euclidean")
    distances = np.ascontiguousarray(distances, dtype=np.float32)
    np.fill_diagonal(distances, 0.0)
    return labels, distances


def measure_configurations(function, warmups: int, repeats: int):
    for _ in range(warmups):
        function()
    runs = {}
    scores = {}
    for _ in range(repeats):
        results = function()
        for name, result in results.items():
            scores[name] = result["scores"]
            runs.setdefault(name, []).append({kernel: float(value) for kernel, value in result["kernel_times"].items()})
    return scores, runs


def summarise_runs(runs):
    summary = {}
    for name, configuration_runs in runs.items():
        kernel_names = list(configuration_runs[0])
        kernel_means = {kernel: float(np.mean([run[kernel] for run in configuration_runs])) for kernel in kernel_names}
        kernel_stddevs = {kernel: float(np.std([run[kernel] for run in configuration_runs])) for kernel in kernel_names}
        totals = [sum(run.values()) for run in configuration_runs]
        summary[name] = {"kernel_means": kernel_means, "kernel_stddevs": kernel_stddevs,
                         "total_mean": float(np.mean(totals)), "total_stddev": float(np.std(totals))}
    return summary


def print_summary(summary):
    print("CUDA event timings (mean ± stddev, milliseconds):")
    print("Totals are the sum of six timed kernel launches; allocations and copies are excluded.")
    for name, result in summary.items():
        print(f"  {name}: total {result['total_mean']:.4f} ± {result['total_stddev']:.4f} ms")
        mean_sum = sum(result["kernel_means"].values())
        for kernel, milliseconds in sorted(result["kernel_means"].items(), key=lambda item: item[1], reverse=True):
            share = 100.0 * milliseconds / mean_sum if mean_sum else 0.0
            print(f"    {kernel:<30} {milliseconds:9.4f} ± {result['kernel_stddevs'][kernel]:.4f} ms ({share:5.1f}%)")


def write_report(path: Path, args, runs, summary, max_errors):
    report = {"samples": args.samples, "clusters": args.clusters, "features": args.features,
              "seed": args.seed, "units": "milliseconds", "kernel_runs": runs,
              "benchmark_results": summary, "max_absolute_score_errors": max_errors}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
