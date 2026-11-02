"""Comparison of four ensemble strategies for every trade-off weight alpha.

1. ``full_ensemble``       all models, majority vote / mean
2. ``optimized_ensemble``  pruned subset S_alpha, majority vote / mean
3. ``full_stacked``        stacking meta-learner on all models
4. ``optimized_stacked``   stacking meta-learner on S_alpha

The ``diversity`` column always refers to the base models that are combined, and
``fitness = alpha * P~ + (1 - alpha) * D~`` uses the [0, 1] performance score P and
the diversity D rescaled with the same :class:`~prunestack.metrics.fitness.FitnessScaler`
as the optimization.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from prunestack.ensemble.stacking import StackedEnsemble
from prunestack.metrics.diversity import average_diversity
from prunestack.metrics.fitness import IDENTITY_SCALER, FitnessScaler
from prunestack.metrics.performance import (
    evaluate_ensemble_performance,
    normalize_performance_score,
    resolve_metric,
)

logger = logging.getLogger(__name__)


def compare_ensemble_strategies(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    optimized_subsets: dict[float, list[int]],
    task_type: str,
    cv_folds: int = 5,
    seed: int = 42,
    performance_metric: str | None = None,
    diversity_metric: str = "auto",
    meta_learner: str | None = None,
    meta_learner_params: dict[str, Any] | None = None,
    base_probabilities: np.ndarray | None = None,
    scaler: FitnessScaler | None = None,
) -> pd.DataFrame:
    """Evaluate the four strategies for each alpha in ``optimized_subsets``.

    Parameters
    ----------
    optimized_subsets : dict[float, list[int]]
        alpha -> model indices selected by the optimizer. If empty, the full pool
        is used with alpha = 0.5.
    meta_learner, meta_learner_params
        Stacking meta-learner (see :mod:`prunestack.ensemble.stacking`).
    base_probabilities : ndarray (n_samples, n_models, n_classes) or None
        Out-of-fold class probabilities used as stacking inputs (classification).
    scaler : FitnessScaler or None
        Rescaling of P and D in the ``fitness`` column (``None`` = raw values).

    Returns
    -------
    DataFrame with columns ``strategy_name, alpha, n_models, performance, diversity, fitness``.
    """
    y_true = np.asarray(y_true)
    predictions_matrix = np.asarray(predictions_matrix, dtype=float)
    n_models = predictions_matrix.shape[1]
    all_indices = list(range(n_models))
    metric = resolve_metric(performance_metric, task_type)
    subsets = optimized_subsets or {0.5: all_indices}
    scaler = scaler or IDENTITY_SCALER

    def performance(indices: list[int]) -> float:
        return evaluate_ensemble_performance(
            y_true, predictions_matrix, indices, task_type, performance_metric=metric,
        )

    def diversity(indices: list[int]) -> float:
        return average_diversity(
            predictions_matrix, indices, task_type, y_true, diversity_metric=diversity_metric,
        )

    logger.info("Training full stacked ensemble for comparison...")
    stacker = StackedEnsemble(
        base_predictions_matrix=predictions_matrix,
        y_true=y_true,
        task_type=task_type,
        cv_folds=cv_folds,
        meta_learner=meta_learner,
        meta_learner_params=meta_learner_params,
        seed=seed,
        base_probabilities=base_probabilities,
    )
    stacker.train()

    full_perf = performance(all_indices)
    full_div = diversity(all_indices)
    full_stacked_perf = stacker.evaluate(performance_metric=metric)

    rows: list[dict[str, Any]] = []
    for alpha in sorted(subsets):
        subset = list(subsets[alpha])
        subset_div = diversity(subset)
        candidates = [
            ("full_ensemble", n_models, full_perf, full_div),
            ("optimized_ensemble", len(subset), performance(subset), subset_div),
            ("full_stacked", n_models, full_stacked_perf, full_div),
            ("optimized_stacked", len(subset), stacker.evaluate(subset, metric), subset_div),
        ]
        for name, size, perf, div in candidates:
            perf_score = normalize_performance_score(perf, metric, y_true)
            rows.append({
                "strategy_name": name,
                "alpha": float(alpha),
                "n_models": size,
                "performance": perf,
                "diversity": div,
                "fitness": scaler.combine(alpha, perf_score, div),
            })

    df = pd.DataFrame(rows)
    logger.info("Strategy comparison complete: %d rows across %d alpha values", len(df), len(subsets))
    return df
