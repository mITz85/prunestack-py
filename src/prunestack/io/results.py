"""Experiment results aggregation, saving, and loading.

Provides :class:`ExperimentResults` (dataclass) and save/load functions
for persisting full experiment outputs in an organized directory structure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@dataclass
class ExperimentResults:
    """Container for all experiment results.

    Attributes
    ----------
    dataset_info : dict
        Dataset metadata (path, task_type, n_samples, etc.).
    base_models_performance : DataFrame or None
        Per-model, per-fold cross-validated metrics of the base-model pool.
    optimization_results : DataFrame or None
        One row per alpha with the subset selected by the optimizer.
    stacking_results : DataFrame or None
        Stacking performance on the full pool and on each selected subset.
    strategy_comparison : DataFrame or None
        Comparison of ensemble combination strategies.
    meta_features : dict
        Dataset-, pool- and ensemble-level meta-features (name -> value); the
        inputs of the meta-learner.
    optimization_summary : dict
        Summary of the EHO results over the alpha grid. Derived from the
        optimization outcome, so it is stored apart from the meta-features
        (using it as meta-learner input would be target leakage).
    execution_metadata : dict
        Package version, seeds, full configuration and timings.
    """

    dataset_info: dict = field(default_factory=dict)
    base_models_performance: pd.DataFrame | None = None
    optimization_results: pd.DataFrame | None = None
    stacking_results: pd.DataFrame | None = None
    strategy_comparison: pd.DataFrame | None = None
    meta_features: dict = field(default_factory=dict)
    optimization_summary: dict = field(default_factory=dict)
    execution_metadata: dict = field(default_factory=dict)


def save_results(results: ExperimentResults, output_dir: str | Path) -> Path:
    """Save experiment results to an organized directory structure.

    Structure:
        output_dir/
          base_models/
            base_models_performance.csv
          optimization/
            optimization_results.csv
            optimization_summary.json   (not a meta-feature: target leakage)
          stacking/
            stacking_results.csv
          comparison/
            strategy_comparison.csv
          meta_features.json
          experiment_summary.json

    Returns
    -------
    Path
        The output directory path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base models
    bm_dir = output_dir / "base_models"
    bm_dir.mkdir(exist_ok=True)
    if results.base_models_performance is not None:
        results.base_models_performance.to_csv(
            bm_dir / "base_models_performance.csv", index=False
        )
        logger.info("Saved base_models_performance.csv")

    # Optimization
    opt_dir = output_dir / "optimization"
    opt_dir.mkdir(exist_ok=True)
    if results.optimization_results is not None:
        results.optimization_results.to_csv(
            opt_dir / "optimization_results.csv", index=False
        )
        logger.info("Saved optimization_results.csv")

    if results.optimization_summary:
        with open(opt_dir / "optimization_summary.json", "w", encoding="utf-8") as f:
            json.dump(results.optimization_summary, f, indent=2, cls=_NumpyEncoder)
        logger.info("Saved optimization_summary.json")

    # Stacking
    stack_dir = output_dir / "stacking"
    stack_dir.mkdir(exist_ok=True)
    if results.stacking_results is not None:
        results.stacking_results.to_csv(
            stack_dir / "stacking_results.csv", index=False
        )
        logger.info("Saved stacking_results.csv")

    # Strategy comparison
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(exist_ok=True)
    if results.strategy_comparison is not None:
        results.strategy_comparison.to_csv(
            comp_dir / "strategy_comparison.csv", index=False
        )
        logger.info("Saved strategy_comparison.csv")

    # Meta-features
    with open(output_dir / "meta_features.json", "w", encoding="utf-8") as f:
        json.dump(results.meta_features, f, indent=2, cls=_NumpyEncoder)
    logger.info("Saved meta_features.json (%d features)", len(results.meta_features))

    # Experiment summary
    summary = {
        "dataset_info": results.dataset_info,
        "execution_metadata": results.execution_metadata,
        "n_meta_features": len(results.meta_features),
        "has_base_models": results.base_models_performance is not None,
        "has_optimization": results.optimization_results is not None,
        "has_stacking": results.stacking_results is not None,
        "has_comparison": results.strategy_comparison is not None,
    }
    with open(output_dir / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)
    logger.info("Saved experiment_summary.json")

    logger.info("All results saved to %s", output_dir)
    return output_dir


def load_results(output_dir: str | Path) -> ExperimentResults:
    """Load experiment results from a saved directory.

    Parameters
    ----------
    output_dir : str or Path
        Directory containing saved results.

    Returns
    -------
    ExperimentResults
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {output_dir}")

    results = ExperimentResults()

    # Load summary
    summary_path = output_dir / "experiment_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        results.dataset_info = summary.get("dataset_info", {})
        results.execution_metadata = summary.get("execution_metadata", {})

    # Load base models
    bm_path = output_dir / "base_models" / "base_models_performance.csv"
    if bm_path.exists():
        results.base_models_performance = pd.read_csv(bm_path)

    # Load optimization
    opt_path = output_dir / "optimization" / "optimization_results.csv"
    if opt_path.exists():
        results.optimization_results = pd.read_csv(opt_path)

    # Load stacking
    stack_path = output_dir / "stacking" / "stacking_results.csv"
    if stack_path.exists():
        results.stacking_results = pd.read_csv(stack_path)

    # Load comparison
    comp_path = output_dir / "comparison" / "strategy_comparison.csv"
    if comp_path.exists():
        results.strategy_comparison = pd.read_csv(comp_path)

    summary_path = output_dir / "optimization" / "optimization_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            results.optimization_summary = json.load(f)

    # Load meta-features
    mf_path = output_dir / "meta_features.json"
    if mf_path.exists():
        with open(mf_path, encoding="utf-8") as f:
            results.meta_features = json.load(f)

    logger.info("Loaded results from %s", output_dir)
    return results
