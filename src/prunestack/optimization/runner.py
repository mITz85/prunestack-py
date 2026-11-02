"""Run the subset optimizer for one trade-off weight alpha or for an alpha grid."""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from prunestack.metrics.fitness import FitnessScaler, evaluate_subset
from prunestack.optimization.objective import FitnessObjective
from prunestack.optimization.protocol import OptimizerProtocol, OptimizationResult

logger = logging.getLogger(__name__)


def optimize_ensemble(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    alpha: float,
    task_type: str,
    optimizer: OptimizerProtocol,
    seed: int = 42,
    performance_metric: str | None = None,
    diversity_metric: str = "auto",
    scaler: FitnessScaler | None = None,
) -> OptimizationResult:
    """Find the model subset maximising α·P̃(S) + (1 − α)·D̃(S).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels / target values.
    predictions_matrix : np.ndarray
        OOF predictions, shape (n_samples, n_models).
    alpha : float
        Performance weight in fitness function (diversity weight is 1−α).
    task_type : str
        'classification' or 'regression'.
    optimizer : OptimizerProtocol
        Optimizer instance (e.g., EHOOptimizer).
    seed : int
        Random seed for reproducibility.
    performance_metric : str or None
        Named performance metric (task default if None).
    diversity_metric : str
        Diversity measure (``'auto'``: disagreement / ambiguity).
    scaler : FitnessScaler or None
        Rescaling of P and D before mixing (``None`` = raw values).

    Returns
    -------
    OptimizationResult
    """
    n_models = predictions_matrix.shape[1]

    # Create objective function
    objective = FitnessObjective(
        y_true=y_true,
        predictions_matrix=predictions_matrix,
        alpha=alpha,
        task_type=task_type,
        maximize=True,
        performance_metric=performance_metric,
        diversity_metric=diversity_metric,
        scaler=scaler,
    )

    logger.info(
        "Optimizing ensemble: alpha=%.2f, n_models=%d, seed=%d", alpha, n_models, seed,
    )

    result = optimizer.optimize(
        objective_function=objective,
        n_models=n_models,
        seed=seed,
    )

    logger.info(
        "Optimization complete: fitness=%.6f, selected=%d/%d, "
        "cache_hit_rate=%.1f%%, %d evaluations",
        result.best_fitness,
        result.n_selected,
        n_models,
        objective.cache_hit_rate * 100,
        objective.n_calls,
    )

    # Add objective stats to metadata
    result.metadata["objective_calls"] = objective.n_calls
    result.metadata["cache_hits"] = objective.n_cache_hits
    result.metadata["cache_size"] = objective.cache_size

    return result


def optimize_alpha_grid(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    alpha_grid: list[float],
    *,
    optimizer: OptimizerProtocol,
    task_type: str,
    seed: int = 42,
    performance_metric: str | None = None,
    diversity_metric: str = "auto",
    scaler: FitnessScaler | None = None,
) -> pd.DataFrame:
    """Run one optimization per trade-off weight alpha.

    Returns
    -------
    pd.DataFrame
        One row per alpha with columns ``alpha, best_fitness, best_subset_indices,
        n_models_selected, performance, performance_score, diversity, n_evaluations,
        n_iterations, execution_time, convergence_history``. ``performance`` is the raw
        metric of the selected subset, ``performance_score`` its [0, 1] score P and
        ``diversity`` the raw D (before rescaling). ``best_subset_indices`` and
        ``convergence_history`` are JSON-encoded lists.
    """
    alphas = [float(a) for a in alpha_grid]
    logger.info("Starting alpha-grid optimization: %d values", len(alphas))
    start_time = time.time()

    alpha_iter = alphas
    if tqdm is not None:
        alpha_iter = tqdm(alphas, desc="Alpha-grid optimization", unit="alpha")

    rows: list[dict] = []
    for alpha in alpha_iter:
        result = optimize_ensemble(
            y_true=y_true,
            predictions_matrix=predictions_matrix,
            alpha=alpha,
            task_type=task_type,
            optimizer=optimizer,
            seed=seed,
            performance_metric=performance_metric,
            diversity_metric=diversity_metric,
            scaler=scaler,
        )
        details = evaluate_subset(
            y_true, predictions_matrix, result.best_indices, alpha, task_type,
            performance_metric=performance_metric, diversity_metric=diversity_metric,
            scaler=scaler, return_predictions=False,
        )
        rows.append({
            "alpha": alpha,
            "best_fitness": result.best_fitness,
            "best_subset_indices": json.dumps([int(i) for i in result.best_indices]),
            "n_models_selected": result.n_selected,
            "performance": details["performance"],
            "performance_score": details["performance_score"],
            "diversity": details["diversity"],
            "n_evaluations": result.metadata.get("n_evaluations", result.metadata["objective_calls"]),
            "n_iterations": result.n_iterations,
            "execution_time": round(result.execution_time, 4),
            "convergence_history": json.dumps([float(v) for v in result.convergence_history]),
        })

    logger.info(
        "Alpha-grid optimization completed: %d values in %.2fs",
        len(alphas), time.time() - start_time,
    )
    return pd.DataFrame(rows)
