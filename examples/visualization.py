#!/usr/bin/env python3
"""Rebuild the HTML report and plots from a saved results directory.

Usage:
    python examples/visualization.py /path/to/dataset/results
"""

from __future__ import annotations

import sys
from pathlib import Path

from prunestack.io.results import load_results
from prunestack.io.visualization import generate_experiment_report


def main(results_dir: str) -> None:
    results = load_results(results_dir)
    report = generate_experiment_report(results, Path(results_dir) / "experiment_report.html")
    print(f"Report: {report}")
    print(f"Plots:  {Path(results_dir) / 'plots'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python examples/visualization.py /path/to/dataset/results")
    main(sys.argv[1])
