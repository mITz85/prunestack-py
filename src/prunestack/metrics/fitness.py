"""Fitness function for ensemble pruning.

Fitness(S) = α · P̃(S) + (1 − α) · D̃(S)

P(S) is the [0, 1] performance score of the plain ensemble of S (macro-F1, or a
skill score for error metrics) and D(S) its diversity. Because P and D occupy
very different parts of [0, 1] (e.g. macro-F1 ≈ 0.75 vs. disagreement ≈ 0.25),
both are rescaled with a :class:`FitnessScaler` before they are mixed, so that
α weights comparable quantities:

    P̃(S) = (P(S) − P_lo) / (P_hi − P_lo),   D̃(S) = (D(S) − D_lo) / (D_hi − D_lo)

The bounds are robust quantiles (1st/99th percentile by default) of P and D over
a fixed reference sample of random subsets whose size is uniform on {2, ..., M}
(:func:`fit_fitness_scaler`), so small and large subsets are equally represented. The map
is affine and *not* clipped, so it never changes the ranking of subsets for a
given α and an optimizer can still improve beyond the reference range.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from prunestack.metrics.diversity import average_diversity
from prunestack.metrics.performance import (
    ensemble_predictions_classification,
    ensemble_predictions_regression,
    evaluate_ensemble_performance,
    normalize_performance_score,
    resolve_metric,
)

logger = logging.getLogger(__name__)

VALID_NORMALIZATIONS = ("minmax", "none")


@dataclass(frozen=True)
class FitnessScaler:
    """Affine rescaling of P(S) and D(S) before they are combined.

    The default instance is the identity (bounds ``(0, 1)``), i.e. the raw
    fitness ``α·P + (1 − α)·D``.
    """

    performance_bounds: tuple[float, float] = (0.0, 1.0)
    diversity_bounds: tuple[float, float] = (0.0, 1.0)
    method: str = "none"
    n_reference_subsets: int = 0
    quantiles: tuple[float, float] = (0.0, 1.0)

    @staticmethod
    def _scale(value: float, bounds: tuple[float, float]) -> float:
        lo, hi = bounds
        span = hi - lo
        return (value - lo) / span if span > 1e-12 else value - lo

    def scale_performance(self, performance_score: float) -> float:
        return self._scale(performance_score, self.performance_bounds)

    def scale_diversity(self, diversity: float) -> float:
        return self._scale(diversity, self.diversity_bounds)

    def combine(self, alpha: float, performance_score: float, diversity: float) -> float:
        """α · P̃ + (1 − α) · D̃."""
        return float(alpha * self.scale_performance(performance_score)
                     + (1.0 - alpha) * self.scale_diversity(diversity))

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "performance_bounds": list(self.performance_bounds),
            "diversity_bounds": list(self.diversity_bounds),
            "n_reference_subsets": self.n_reference_subsets,
            "quantiles": list(self.quantiles),
        }


IDENTITY_SCALER = FitnessScaler()


def sample_random_subsets(
    rng: np.random.Generator,
    n_subsets: int,
    n_models: int,
    inclusion_range: tuple[float, float],
    min_size: int = 1,
) -> np.ndarray:
    """Random binary subsets of varied size.

    Each subset first draws its own inclusion probability p ~ U(inclusion_range),
    then every model is included independently with probability p. Subsets with
    fewer than ``min_size`` models are topped up with randomly chosen models.

    Returns
    -------
    ndarray of shape (n_subsets, n_models) with values in {0, 1}.
    """
    lo, hi = inclusion_range
    p = rng.uniform(lo, hi, size=(n_subsets, 1))
    subsets = (rng.random((n_subsets, n_models)) < p).astype(np.int8)
    min_size = min(min_size, n_models)
    for row in subsets:
        missing = min_size - int(row.sum())
        if missing > 0:
            row[rng.choice(np.flatnonzero(row == 0), size=missing, replace=False)] = 1
    return subsets


def sample_uniform_size_subsets(
    rng: np.random.Generator,
    n_subsets: int,
    n_models: int,
    min_size: int = 2,
) -> np.ndarray:
    """Random binary subsets whose size k is uniform on {min_size, ..., n_models}.

    The k models of each subset are drawn uniformly without replacement. Unlike
    :func:`sample_random_subsets` (binomial sizes concentrated around p·M), every subset
    size the optimizer can reach is represented, so the reference range of P and D also
    covers the small subsets that pruning tends to select.

    Returns
    -------
    ndarray of shape (n_subsets, n_models) with values in {0, 1}.
    """
    min_size = max(1, min(min_size, n_models))
    sizes = rng.integers(min_size, n_models + 1, size=n_subsets)
    subsets = np.zeros((n_subsets, n_models), dtype=np.int8)
    for row, k in zip(subsets, sizes):
        row[rng.choice(n_models, size=k, replace=False)] = 1
    return subsets


def _performance_and_diversity(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    task_type: str,
    metric: str,
    diversity_metric: str,
) -> tuple[float, float, float]:
    """Raw performance, its [0, 1] score and diversity of one subset."""
    raw = evaluate_ensemble_performance(
        y_true, predictions_matrix, subset_indices, task_type, performance_metric=metric,
    )
    div = average_diversity(
        predictions_matrix, subset_indices, task_type, y_true, diversity_metric=diversity_metric,
    )
    return raw, normalize_performance_score(raw, metric, y_true), div


def fit_fitness_scaler(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    task_type: str,
    performance_metric: str | None = None,
    diversity_metric: str = "auto",
    method: str = "minmax",
    n_subsets: int = 500,
    seed: int = 1985,
    quantiles: tuple[float, float] = (0.01, 0.99),
) -> FitnessScaler:
    """Estimate the P/D bounds of the fitness from a reference sample of subsets.

    The reference sample depends only on the data and ``seed`` (use the
    experiment seed, not the optimizer seed), so all optimization runs, all α
    values and all optimizers of a dataset share one fitness scale.

    Parameters
    ----------
    method : 'minmax' | 'none'
        ``'none'`` returns :data:`IDENTITY_SCALER`.
    n_subsets : int
        Number of random reference subsets; sizes uniform on {2, ..., M}
        (see :func:`sample_uniform_size_subsets`).
    quantiles : (low, high)
        Quantiles of the reference distribution used as bounds (robust min/max).
    """
    if method not in VALID_NORMALIZATIONS:
        raise ValueError(f"method must be one of {VALID_NORMALIZATIONS}, got {method!r}")
    if method == "none":
        return IDENTITY_SCALER
    if n_subsets < 2:
        raise ValueError(f"n_subsets must be >= 2, got {n_subsets}")
    q_lo, q_hi = quantiles
    if not 0.0 <= q_lo < q_hi <= 1.0:
        raise ValueError(f"quantiles must satisfy 0 <= low < high <= 1, got {quantiles}")

    predictions_matrix = np.asarray(predictions_matrix, dtype=float)
    y_true = np.asarray(y_true)
    metric = resolve_metric(performance_metric, task_type)
    n_models = predictions_matrix.shape[1]
    rng = np.random.default_rng(seed)
    subsets = sample_uniform_size_subsets(rng, n_subsets, n_models, min_size=2)

    perf = np.empty(n_subsets)
    div = np.empty(n_subsets)
    for i, row in enumerate(subsets):
        _, perf[i], div[i] = _performance_and_diversity(
            y_true, predictions_matrix, np.flatnonzero(row), task_type, metric, diversity_metric,
        )
    scaler = FitnessScaler(
        performance_bounds=(float(np.quantile(perf, q_lo)), float(np.quantile(perf, q_hi))),
        diversity_bounds=(float(np.quantile(div, q_lo)), float(np.quantile(div, q_hi))),
        method=method,
        n_reference_subsets=n_subsets,
        quantiles=(float(q_lo), float(q_hi)),
    )
    logger.info(
        "Fitness normalization (%s, %d reference subsets): P in [%.4f, %.4f], D in [%.4f, %.4f]",
        method, n_subsets, *scaler.performance_bounds, *scaler.diversity_bounds,
    )
    return scaler


def calculate_fitness(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    alpha: float,
    task_type: str,
    performance_metric: str | None = None,
    diversity_metric: str = "auto",
    scaler: FitnessScaler | None = None,
) -> float:
    """Fitness α · P̃(S) + (1 − α) · D̃(S) of one subset (higher is better).

    Parameters
    ----------
    y_true : ndarray
        True labels / target values.
    predictions_matrix : ndarray (n_samples, n_models)
    subset_indices : array-like of int
        Model indices to evaluate.
    alpha : float in [0, 1]
        Weight for performance; (1 - alpha) is the weight for diversity.
    task_type : 'classification' | 'regression'
    performance_metric : str or None
        Named metric. If None, uses the task-type default (macro-F1 / RMSE).
        Error metrics are mapped to a skill score
        (:func:`~prunestack.metrics.performance.normalize_performance_score`).
    diversity_metric : str
        ``'auto'`` uses disagreement for classification and normalised
        ambiguity (Ā/Ē) for regression.
    scaler : FitnessScaler or None
        Rescaling of P and D; ``None`` means the identity (raw P and D).
    """
    return evaluate_subset(
        y_true, predictions_matrix, subset_indices, alpha, task_type,
        performance_metric=performance_metric, diversity_metric=diversity_metric,
        scaler=scaler, return_predictions=False,
    )["fitness"]


def evaluate_subset(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    alpha: float,
    task_type: str,
    performance_metric: str | None = None,
    diversity_metric: str = "auto",
    scaler: FitnessScaler | None = None,
    return_predictions: bool = True,
) -> dict:
    """Evaluate a model subset and return detailed results.

    Returns
    -------
    dict with keys:
        fitness : float               α · P̃ + (1 − α) · D̃
        performance : float           raw metric value (for reporting)
        performance_score : float     [0, 1] performance score P
        performance_scaled : float    P̃
        diversity : float             raw diversity D
        diversity_scaled : float      D̃
        alpha : float
        ensemble_predictions : ndarray (only if ``return_predictions``)
    """
    scaler = scaler or IDENTITY_SCALER
    metric = resolve_metric(performance_metric, task_type)
    raw, perf_score, div = _performance_and_diversity(
        np.asarray(y_true), predictions_matrix, subset_indices, task_type, metric, diversity_metric,
    )
    result = {
        "fitness": scaler.combine(alpha, perf_score, div),
        "performance": float(raw),
        "performance_score": float(perf_score),
        "performance_scaled": scaler.scale_performance(perf_score),
        "diversity": float(div),
        "diversity_scaled": scaler.scale_diversity(div),
        "alpha": float(alpha),
    }
    if return_predictions:
        combine = (ensemble_predictions_classification if task_type == "classification"
                   else ensemble_predictions_regression)
        result["ensemble_predictions"] = combine(predictions_matrix, subset_indices)
    return result
