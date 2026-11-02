#!/usr/bin/env python3
"""Run the full pipeline on one dataset and print a short summary.

Usage:
    python examples/basic_usage.py /path/to/dataset
"""

from __future__ import annotations

import sys

from prunestack.config import ExperimentConfig
from prunestack.experiment.runner import ExperimentRunner
from prunestack.utils.logging import setup_logging


def main(dataset_path: str) -> None:
    setup_logging()
    config = ExperimentConfig(dataset_path=dataset_path)  # protocol defaults (seed 1985, 6 x 20 models)
    runner = ExperimentRunner(config)  # writes to <dataset_path>/results
    results = runner.run_full_experiment()

    info = results.dataset_info
    print(f"\n{info['task_type']}: {info['n_samples']} samples, {info['n_features']} features, "
          f"{info['n_models']} base models")

    if results.optimization_results is not None:
        print("\nSelected subset size per alpha:")
        for _, row in results.optimization_results.iterrows():
            print(f"  alpha={row['alpha']:.1f}  models={row['n_models_selected']:3d}  "
                  f"fitness={row['best_fitness']:.4f}")

    if results.strategy_comparison is not None:
        print("\nStrategy comparison (mean over alpha):")
        summary = results.strategy_comparison.groupby("strategy_name")["performance"].mean()
        for name, performance in summary.items():
            print(f"  {name:20s} {performance:.4f}")

    print(f"\n{len(results.meta_features)} meta-features; report: {runner.output_dir / 'experiment_report.html'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python examples/basic_usage.py /path/to/dataset")
    main(sys.argv[1])
