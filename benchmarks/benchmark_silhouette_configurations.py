"""Compare every compile-time silhouette kernel configuration on identical input."""

import argparse
from pathlib import Path

import frogml
import numpy as np

from silhouette_benchmark_common import make_data, measure_configurations, print_summary, summarise_runs, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--features", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timing-output", type=Path, default=Path("build/silhouette-configuration-timings.json"))
    args = parser.parse_args()
    if args.samples < 2 or not 2 <= args.clusters < args.samples:
        parser.error("require 2 <= --clusters < --samples")
    if args.features < 1 or args.warmups < 0 or args.repeats < 1:
        parser.error("features and repeats must be positive; warmups cannot be negative")

    labels, distances = make_data(args.samples, args.clusters, args.features, args.seed)
    scores, runs = measure_configurations(
        lambda: frogml.silhouette_samples_bench_configurations(labels, distances, args.clusters),
        args.warmups, args.repeats)
    baseline_scores = scores["baseline_a_tile_32"]
    max_errors = {}
    for name, scores_for_configuration in scores.items():
        np.testing.assert_allclose(scores_for_configuration, baseline_scores, rtol=1e-4, atol=1e-4)
        max_errors[name] = float(np.max(np.abs(scores_for_configuration - baseline_scores)))
    summary = summarise_runs(runs)
    print_summary(summary)
    write_report(args.timing_output, args, runs, summary, max_errors)
    print(f"Saved raw kernel timings to {args.timing_output}")


if __name__ == "__main__":
    main()
