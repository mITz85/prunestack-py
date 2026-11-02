#!/usr/bin/env python3
"""Compare stacking meta-learners on the saved base-model predictions of one experiment.

Requires a finished run (``<dataset>/results/base_models/base_models_predictions.csv``).

Usage:
    python examples/stacking_only.py /path/to/dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from prunestack.data.loader import load_dataset
from prunestack.ensemble.stacking import StackedEnsemble, available_meta_learners


def main(dataset_path: str) -> None:
    predictions_path = Path(dataset_path) / "results" / "base_models" / "base_models_predictions.csv"
    if not predictions_path.exists():
        sys.exit(f"{predictions_path} not found. Run `prunestack --dataset-path {dataset_path}` first.")

    predictions = pd.read_csv(predictions_path).to_numpy(dtype=float)
    bundle = load_dataset(dataset_path, seed=1985)
    y_true = np.asarray(bundle.y)
    metric = "macro-F1" if bundle.task_type == "classification" else "RMSE"
    print(f"{bundle.task_type}: {bundle.n_samples} samples, {predictions.shape[1]} base models\n")

    for meta_learner in available_meta_learners(bundle.task_type):
        stacker = StackedEnsemble(predictions, y_true, bundle.task_type, meta_learner=meta_learner, seed=1985)
        stacker.train()
        print(f"  {meta_learner:24s} {metric} = {stacker.evaluate():.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python examples/stacking_only.py /path/to/dataset")
    main(sys.argv[1])
