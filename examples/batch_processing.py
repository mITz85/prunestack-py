#!/usr/bin/env python3
"""Run the pipeline on every dataset directory below a root directory.

Each subdirectory must contain ``X_test.csv``, ``y_test.csv`` and ``metrics.csv``.
The command-line equivalent for one dataset is ``prunestack --dataset-path <dir>``.

Usage:
    python examples/batch_processing.py /path/to/datasets/
"""

from __future__ import annotations

import sys
from pathlib import Path

from prunestack.config import ExperimentConfig
from prunestack.experiment.runner import ExperimentRunner
from prunestack.utils.logging import setup_logging

REQUIRED_FILES = ("X_test.csv", "y_test.csv", "metrics.csv")


def main(root: str) -> None:
    setup_logging(quiet=True)
    dataset_dirs = sorted(
        d for d in Path(root).iterdir()
        if d.is_dir() and all((d / name).exists() for name in REQUIRED_FILES)
    )
    if not dataset_dirs:
        sys.exit(f"No dataset directories found in {root}")

    failures: dict[str, str] = {}
    for dataset_dir in dataset_dirs:
        print(f"[{dataset_dir.name}] running...")
        try:
            ExperimentRunner(ExperimentConfig(dataset_path=dataset_dir)).run_full_experiment()
        except Exception as e:  # keep going with the remaining datasets
            failures[dataset_dir.name] = str(e)
            print(f"[{dataset_dir.name}] FAILED: {e}")
        else:
            print(f"[{dataset_dir.name}] done")

    print(f"\n{len(dataset_dirs) - len(failures)}/{len(dataset_dirs)} datasets completed")
    for name, error in failures.items():
        print(f"  {name}: {error}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python examples/batch_processing.py /path/to/datasets/")
    main(sys.argv[1])
