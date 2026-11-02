#!/usr/bin/env python3
"""Run the pipeline with a custom alpha grid and EHO settings.

Usage:
    python examples/custom_optimization.py /path/to/dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

from prunestack.config import ExperimentConfig
from prunestack.experiment.runner import ExperimentRunner
from prunestack.utils.logging import setup_logging


def main(dataset_path: str) -> None:
    setup_logging(verbose=True)
    config = ExperimentConfig(
        dataset_path=dataset_path,
        alpha_grid=[round(0.05 * i, 2) for i in range(21)],  # finer than the default 0.1 step
        eho_n_clans=8,
        eho_clan_size=15,
        eho_max_iterations=100,
        eho_alpha=0.5,          # alpha_EHO: pull towards the clan matriarch
        eho_beta=0.1,           # beta_EHO: pull of the matriarch towards the clan centre
        eho_early_stop_patience=20,
        stacking_method="hist_gradient_boosting",  # instead of the NNLS default
    )
    results = ExperimentRunner(config, output_dir=Path(dataset_path) / "results_custom").run_full_experiment()

    optimization = results.optimization_results
    best = optimization.loc[optimization["best_fitness"].idxmax()]
    print(f"\nHighest fitness at alpha={best['alpha']:.2f}: {best['best_fitness']:.4f} "
          f"({best['n_models_selected']} models)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python examples/custom_optimization.py /path/to/dataset")
    main(sys.argv[1])
