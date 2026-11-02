"""Performance metrics and plain ensemble aggregation.

Classification: accuracy, macro-F1 (default). Regression: MAE, MSE, RMSE (default), R².
The pruning objective combines members by majority vote (classification) or by the
mean (regression); see :mod:`prunestack.ensemble.stacking` for the trained combiner.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from scipy.stats import mode
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)

# Metrics where *lower* is better (errors); all others are higher-is-better.
_LOWER_IS_BETTER = {"mae", "mse", "rmse"}

# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return classification accuracy in [0, 1]."""
    return float(accuracy_score(y_true, y_pred))


def calculate_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "macro",
) -> float:
    """Return F1 score with the given averaging strategy.

    Default is macro-averaging: treats all classes equally regardless of
    support. This is the primary performance metric for classification in
    the ensemble pruning experiment.
    """
    return float(f1_score(y_true, y_pred, average=average, zero_division=0))


def ensemble_predictions_classification(
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
) -> np.ndarray:
    """Aggregate classification predictions via majority vote.

    Parameters
    ----------
    predictions_matrix : ndarray of shape (n_samples, n_models)
        Each column holds one model's predictions.
    subset_indices : array-like of int
        Column indices of the models to include.

    Returns
    -------
    ndarray of shape (n_samples,)
        Majority-vote prediction for each sample.
    """
    subset = predictions_matrix[:, subset_indices].astype(int)
    # Ties are resolved towards the smallest class label (scipy.stats.mode).
    return np.asarray(mode(subset, axis=1, keepdims=False).mode).ravel()


# ---------------------------------------------------------------------------
# Regression metrics
# ---------------------------------------------------------------------------


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Mean Absolute Error."""
    return float(mean_absolute_error(y_true, y_pred))


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Mean Squared Error."""
    return float(mean_squared_error(y_true, y_pred))


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return Root Mean Squared Error."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return R² (coefficient of determination)."""
    return float(r2_score(y_true, y_pred))


def ensemble_predictions_regression(
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
) -> np.ndarray:
    """Aggregate regression predictions by averaging selected models.

    Parameters
    ----------
    predictions_matrix : ndarray of shape (n_samples, n_models)
    subset_indices : array-like of int

    Returns
    -------
    ndarray of shape (n_samples,)
    """
    subset = predictions_matrix[:, subset_indices]
    return np.mean(subset, axis=1)


# ---------------------------------------------------------------------------
# Metric dispatch
# ---------------------------------------------------------------------------

_METRIC_FNS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "accuracy": calculate_accuracy,
    "f1": calculate_f1_score,
    "mae": calculate_mae,
    "mse": calculate_mse,
    "rmse": calculate_rmse,
    "r2": calculate_r2,
}


_CLASSIFICATION_METRICS = {"accuracy", "f1"}
_REGRESSION_METRICS = {"mae", "mse", "rmse", "r2"}


def resolve_default_metric(task_type: str) -> str:
    """Return the default metric: macro-F1 for classification, RMSE for regression."""
    return "f1" if task_type == "classification" else "rmse"


def resolve_metric(performance_metric: str | None, task_type: str) -> str:
    """Resolve the effective metric name.

    If *performance_metric* is ``None``, empty, or ``"auto"`` the
    task-appropriate default is returned.  Otherwise a compatibility
    check ensures the metric matches the task type and a clear error
    is raised on mismatch.
    """
    if not performance_metric or performance_metric == "auto":
        return resolve_default_metric(task_type)

    # Validate metric-task compatibility
    if task_type == "classification" and performance_metric in _REGRESSION_METRICS:
        raise ValueError(
            f"Metric '{performance_metric}' is a regression metric but "
            f"task_type is 'classification'. Use one of "
            f"{sorted(_CLASSIFICATION_METRICS)} or set performance_metric "
            f"to 'auto'."
        )
    if task_type == "regression" and performance_metric in _CLASSIFICATION_METRICS:
        raise ValueError(
            f"Metric '{performance_metric}' is a classification metric but "
            f"task_type is 'regression'. Use one of "
            f"{sorted(_REGRESSION_METRICS)} or set performance_metric "
            f"to 'auto'."
        )
    return performance_metric


def calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    task_type: str | None = None,
) -> float:
    """Calculate a named performance metric.

    Parameters
    ----------
    y_true, y_pred : array-like
    metric : str
        One of: accuracy, f1, mae, mse, rmse, r2.  May also be
        ``"auto"`` or empty/None — in that case *task_type* is used
        to pick the default.
    task_type : str or None
        ``'classification'`` or ``'regression'``.  Required when
        *metric* is ``"auto"`` / None / empty so the default can be
        resolved.  Also used for validation when provided.

    Returns
    -------
    float
    """
    # Resolve auto / None / empty to a concrete metric
    if not metric or metric == "auto":
        if task_type is None:
            raise ValueError(
                "task_type is required when metric is 'auto' or not specified."
            )
        metric = resolve_default_metric(task_type)
    elif task_type is not None:
        # Validate compatibility when task_type is known
        metric = resolve_metric(metric, task_type)

    fn = _METRIC_FNS.get(metric)
    if fn is None:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Supported: {sorted(_METRIC_FNS.keys())}"
        )
    return fn(y_true, y_pred)


def is_lower_better(metric: str) -> bool:
    """Return True if the metric is an error metric (lower is better)."""
    return metric in _LOWER_IS_BETTER


def normalize_performance_score(
    raw_score: float,
    metric: str,
    y_true: np.ndarray,
) -> float:
    """Map a raw metric value onto a [0, 1] fitness score (higher = better).

    This is the single source of truth for turning *any* metric into a
    bounded, maximise-friendly score so that the fitness trade-off
    ``α · P(S) + (1 − α) · D(S)`` is meaningful: both performance
    ``P(S)`` and diversity ``D(S)`` then live on the same [0, 1] scale.

    Normalisation rules
    -------------------
    * ``accuracy``, ``f1``, ``r2`` — already higher-is-better and naturally
      bounded; simply clipped to ``[0, 1]`` (negative R² → 0).
    * ``mae``, ``mse``, ``rmse`` — error metrics (unbounded, lower-is-better).
      Converted to a *skill score* ``1 − error / baseline_error`` where
      ``baseline_error`` is the error of the trivial mean predictor
      (mean-absolute-deviation for MAE, variance for MSE, std for RMSE).
      The result is clipped to ``[0, 1]`` so that a model no better than the
      mean predictor scores 0 and a perfect model scores 1.  Crucially the
      mapping is *monotonic* in the raw error (baseline is constant for a
      given ``y_true``), so subset ranking is preserved while the value is
      now normalised and comparable to the diversity term.

    Parameters
    ----------
    raw_score : float
        The raw metric value as returned by :func:`calculate_metric`.
    metric : str
        Resolved metric name (must be concrete, not ``"auto"``).
    y_true : np.ndarray
        Ground-truth targets (needed to derive the baseline error for
        error metrics).

    Returns
    -------
    float
        Normalised score in ``[0, 1]``, higher is better.
    """
    metric = (metric or "").lower()

    if metric in ("accuracy", "f1", "r2"):
        return float(np.clip(raw_score, 0.0, 1.0))

    y = np.asarray(y_true, dtype=float).ravel()
    if y.size == 0:
        return 0.0
    mean_y = float(np.mean(y))

    if metric == "mae":
        baseline = float(np.mean(np.abs(y - mean_y)))
    elif metric == "mse":
        baseline = float(np.mean((y - mean_y) ** 2))
    elif metric == "rmse":
        baseline = float(np.sqrt(np.mean((y - mean_y) ** 2)))
    else:
        # Unknown metric – assume already in a sensible range.
        return float(raw_score)

    if baseline <= 1e-12:
        # Degenerate target (constant); no meaningful skill score.
        return 0.0
    return float(np.clip(1.0 - raw_score / baseline, 0.0, 1.0))


def evaluate_ensemble_score(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    task_type: str,
    performance_metric: str | None = None,
) -> tuple[float, float]:
    """Return ``(normalised_score, raw_score)`` for an ensemble subset.

    Convenience wrapper used by the fitness layer: computes the raw metric
    via :func:`evaluate_ensemble_performance` and its [0, 1] normalisation
    via :func:`normalize_performance_score` in one call.
    """
    metric = resolve_metric(performance_metric, task_type)
    raw = evaluate_ensemble_performance(
        y_true, predictions_matrix, subset_indices, task_type,
        performance_metric=metric,
    )
    return normalize_performance_score(raw, metric, y_true), raw


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------


def evaluate_ensemble_performance(
    y_true: np.ndarray,
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    task_type: str,
    performance_metric: str | None = None,
) -> float:
    """Evaluate ensemble performance for a subset of models.

    Parameters
    ----------
    y_true : ndarray
    predictions_matrix : ndarray (n_samples, n_models)
    subset_indices : array-like of int
    task_type : str
    performance_metric : str or None
        Metric name; ``None`` or ``"auto"`` selects the task default (macro-F1 / RMSE).

    Returns
    -------
    float
        Raw metric value (see :func:`normalize_performance_score` for the [0, 1] score).
    """
    performance_metric = resolve_metric(performance_metric, task_type)

    if task_type == "classification":
        y_pred = ensemble_predictions_classification(predictions_matrix, subset_indices)
    elif task_type == "regression":
        y_pred = ensemble_predictions_regression(predictions_matrix, subset_indices)
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")

    return calculate_metric(y_true, y_pred, performance_metric)
