"""Validate the baseline silhouette configuration and report its GPU kernel timings."""

import argparse
from pathlib import Path

import frogml
import numpy as np
from sklearn.metrics import silhouette_samples as sklearn_silhouette_samples

from silhouette_benchmark_common import make_data, measure_configurations, print_summary, summarise_runs, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing-output", type=Path, default=Path("build/silhouette-kernel-timings.json"))
    args = parser.parse_args()
    if args.samples < 2 or not 2 <= args.clusters < args.samples:
        parser.error("require 2 <= --clusters < --samples")
    if args.features < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("features and repeats must be positive; warmups cannot be negative")

    labels, distances = make_data(args.samples, args.clusters, args.features, args.seed)
    expected = sklearn_silhouette_samples(distances, labels, metric="precomputed")
    actual = frogml.silhouette_samples(labels, distances, args.clusters)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)

    def run_baseline():
        scores, kernel_times = frogml.silhouette_samples_bench(labels, distances, args.clusters)
        return {"baseline_a_tile_32": {"scores": scores, "kernel_times": kernel_times}}

    scores, runs = measure_configurations(run_baseline, args.warmups, args.repeats)
    np.testing.assert_allclose(scores["baseline_a_tile_32"], expected, rtol=1e-4, atol=1e-4)
    summary = summarise_runs(runs)
    print_summary(summary)
    max_error = float(np.max(np.abs(scores["baseline_a_tile_32"] - expected)))
    write_report(args.timing_output, args, runs, summary, {"baseline_a_tile_32": max_error})
    print(f"Saved raw kernel timings to {args.timing_output}")


if __name__ == "__main__":
    main()
