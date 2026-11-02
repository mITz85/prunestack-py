"""Convenience wrappers for evaluating model subsets and the full pool."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from prunestack.config import DEFAULT_ALPHA_GRID
from prunestack.metrics.diversity import average_diversity
from prunestack.metrics.fitness import IDENTITY_SCALER, FitnessScaler, evaluate_subset
from prunestack.metrics.performance import (
    evaluate_ensemble_performance,
    normalize_performance_score,
    resolve_metric,
)

logger = logging.getLogger(__name__)


class EnsembleEvaluator:
    """Evaluate model subsets with cached ground truth.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth labels / targets.
    predictions_matrix : ndarray (n_samples, n_models)
        Out-of-fold predictions for each base model.
    task_type : 'classification' | 'regression'
    performance_metric : str or None
        Named metric.  If None, uses the task-type default.
    diversity_metric : str
        Diversity measure (``'auto'``: disagreement for classification,
        ambiguity for regression).
    scaler : FitnessScaler or None
        Rescaling of P and D in the fitness (``None`` = raw values).
    """

    def __init__(
        self,
        y_true: np.ndarray,
        predictions_matrix: np.ndarray,
        task_type: str,
        performance_metric: str | None = None,
        diversity_metric: str = "auto",
        scaler: FitnessScaler | None = None,
    ) -> None:
        self.scaler = scaler or IDENTITY_SCALER
        self.y_true = np.asarray(y_true)
        self.predictions_matrix = np.asarray(predictions_matrix, dtype=float)
        self.task_type = task_type
        self.performance_metric = performance_metric
        self.diversity_metric = diversity_metric
        self.n_samples, self.n_models = self.predictions_matrix.shape

    # ------------------------------------------------------------------
    # Single evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        subset_indices: list[int] | np.ndarray,
        alpha: float,
    ) -> dict:
        """Evaluate a single subset at trade-off weight ``alpha``.

        Returns dict with fitness, performance, diversity, ensemble_predictions.
        """
        return evaluate_subset(
            self.y_true,
            self.predictions_matrix,
            subset_indices,
            alpha,
            self.task_type,
            performance_metric=self.performance_metric,
            diversity_metric=self.diversity_metric,
            scaler=self.scaler,
        )

    # ------------------------------------------------------------------
    # Grid sweep
    # ------------------------------------------------------------------

    def evaluate_alpha_grid(
        self,
        subset_indices: list[int] | np.ndarray,
        alpha_grid: list[float] | None = None,
    ) -> pd.DataFrame:
        """Fitness of one fixed subset across an alpha grid.

        P(S) and D(S) do not depend on alpha, so only the ``fitness`` column varies
        between rows. Returns columns ``alpha, fitness, performance, diversity``.
        """
        if alpha_grid is None:
            alpha_grid = DEFAULT_ALPHA_GRID

        metric = resolve_metric(self.performance_metric, self.task_type)
        perf = evaluate_ensemble_performance(
            self.y_true, self.predictions_matrix, subset_indices, self.task_type,
            performance_metric=metric,
        )
        perf_score = normalize_performance_score(perf, metric, self.y_true)
        div = average_diversity(
            self.predictions_matrix, subset_indices, self.task_type, self.y_true,
            diversity_metric=self.diversity_metric,
        )
        return pd.DataFrame([
            {
                "alpha": float(alpha),
                "fitness": self.scaler.combine(alpha, perf_score, div),
                "performance": perf,
                "diversity": div,
            }
            for alpha in alpha_grid
        ])


# ---------------------------------------------------------------------------
# Baseline helper
# ---------------------------------------------------------------------------


def evaluate_baseline(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    task_type: str,
    performance_metric: str | None = None,
    diversity_metric: str = "auto",
) -> dict:
    """Evaluate the full model pool (all models) as a baseline.

    Returns dict with performance, diversity, and per-model individual scores.
    """
    predictions_matrix = np.asarray(predictions_matrix, dtype=float)
    n_models = predictions_matrix.shape[1]
    all_indices = list(range(n_models))

    perf = evaluate_ensemble_performance(
        y_true, predictions_matrix, all_indices, task_type,
        performance_metric=performance_metric,
    )
    div = average_diversity(
        predictions_matrix, all_indices, task_type, y_true,
        diversity_metric=diversity_metric,
    )

    # Individual model scores
    individual_scores: list[float] = []
    for i in range(n_models):
        individual_scores.append(
            evaluate_ensemble_performance(
                y_true, predictions_matrix, [i], task_type,
                performance_metric=performance_metric,
            )
        )

    return {
        "n_models": n_models,
        "ensemble_performance": perf,
        "ensemble_diversity": div,
        "individual_scores": individual_scores,
        "mean_individual_score": float(np.mean(individual_scores)),
        "best_individual_score": float(np.max(individual_scores)),
        "worst_individual_score": float(np.min(individual_scores)),
    }
