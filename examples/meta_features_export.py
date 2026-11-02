#!/usr/bin/env python3
"""Collect the meta-features of several finished experiments into one table.

Reads ``<dataset>/results/meta_features.json`` (written by the pipeline) for every
dataset below a root directory and writes a single CSV with one row per dataset,
ready to be used as the training set of a meta-learner.

Usage:
    python examples/meta_features_export.py /path/to/datasets/ meta_features.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def main(root: str, output_csv: str) -> None:
    rows = []
    for path in sorted(Path(root).glob("*/results/meta_features.json")):
        features = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"dataset": path.parent.parent.name, **features})
    if not rows:
        sys.exit(f"No results/meta_features.json found below {root}")

    table = pd.DataFrame(rows).set_index("dataset")
    table.to_csv(output_csv)
    print(f"Wrote {table.shape[0]} datasets x {table.shape[1]} meta-features to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python examples/meta_features_export.py /path/to/datasets/ output.csv")
    main(sys.argv[1], sys.argv[2])
