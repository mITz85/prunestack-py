"""Diversity metrics for ensemble evaluation.

Classification: Q-statistic, disagreement measure, double-fault measure.
Regression: ambiguity (Krogh–Vedelsby), correlation coefficient, RMSE difference.
Pairwise diversity matrix and average diversity helpers.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Classification pairwise diversity
# ---------------------------------------------------------------------------


def q_statistic(
    pred_i: np.ndarray,
    pred_j: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Yule's Q-statistic between two classifiers.

    Q = (N11*N00 - N01*N10) / (N11*N00 + N01*N10)

    Where:
        N11 = both correct, N00 = both wrong,
        N10 = i correct j wrong, N01 = i wrong j correct.

    Returns a value in [-1, 1].  Lower Q → more diverse.
    """
    correct_i = (pred_i == y_true).astype(int)
    correct_j = (pred_j == y_true).astype(int)

    n11 = int(np.sum(correct_i & correct_j))
    n00 = int(np.sum((1 - correct_i) & (1 - correct_j)))
    n10 = int(np.sum(correct_i & (1 - correct_j)))
    n01 = int(np.sum((1 - correct_i) & correct_j))

    numerator = n11 * n00 - n01 * n10
    denominator = n11 * n00 + n01 * n10

    if denominator == 0:
        # Perfect agreement or no information – return 0 (neutral)
        return 0.0
    return float(numerator / denominator)


def disagreement_measure(pred_i: np.ndarray, pred_j: np.ndarray) -> float:
    """Proportion of samples where two models disagree.

    Returns a value in [0, 1].  Higher → more diverse.
    """
    return float(np.mean(pred_i != pred_j))


def double_fault_measure(
    pred_i: np.ndarray,
    pred_j: np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Proportion of samples where *both* models are wrong.

    Returns a value in [0, 1].  Lower → better.
    """
    wrong_i = pred_i != y_true
    wrong_j = pred_j != y_true
    return float(np.mean(wrong_i & wrong_j))


# ---------------------------------------------------------------------------
# Regression pairwise diversity
# ---------------------------------------------------------------------------


def correlation_coefficient(pred_i: np.ndarray, pred_j: np.ndarray) -> float:
    """Pearson correlation between two prediction vectors.

    Returns a value in [-1, 1].  Lower absolute value → more diverse.
    """
    if np.std(pred_i) == 0 or np.std(pred_j) == 0:
        return 1.0  # constant predictions → treat as identical
    return float(np.corrcoef(pred_i, pred_j)[0, 1])


def rmse_difference(pred_i: np.ndarray, pred_j: np.ndarray) -> float:
    """Root Mean Squared Difference between two prediction vectors.

    Higher → more diverse (not bounded to [0, 1] on its own).
    """
    return float(np.sqrt(np.mean((pred_i - pred_j) ** 2)))


# ---------------------------------------------------------------------------
# Regression ensemble-level diversity: ambiguity (Krogh & Vedelsby, 1995)
# ---------------------------------------------------------------------------


def ambiguity_decomposition(
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Ambiguity decomposition of an averaging regression ensemble.

    For a subset S of k models with averaging ensemble f̄ = (1/k) Σ_i f_i::

        Ā     = mean_i mean_n (f_i − f̄)²     (average ambiguity)
        Ē     = mean_i mean_n (f_i − y)²     (average member error)
        E_ens = mean_n (f̄ − y)²              (ensemble error)

    which satisfies the identity ``E_ens = Ē − Ā`` (Krogh & Vedelsby, 1995).

    Returns
    -------
    dict with keys ``ambiguity``, ``avg_member_error``, ``ensemble_error``.
    """
    idx = np.asarray(subset_indices, dtype=int)
    sub = np.asarray(predictions_matrix, dtype=float)[:, idx]
    y = np.asarray(y_true, dtype=float).reshape(-1, 1)
    f_bar = sub.mean(axis=1, keepdims=True)
    amb = float(np.mean((sub - f_bar) ** 2))
    avg_err = float(np.mean((sub - y) ** 2))
    ens_err = float(np.mean((f_bar - y) ** 2))
    return {
        "ambiguity": amb,
        "avg_member_error": avg_err,
        "ensemble_error": ens_err,
    }


def ambiguity_diversity(
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    y_true: np.ndarray,
) -> float:
    """Normalised ambiguity diversity Ā / Ē in [0, 1] (higher → more diverse).

    Because ``E_ens = Ē − Ā ≥ 0`` it follows that ``0 ≤ Ā ≤ Ē``, so the ratio
    is bounded in [0, 1].  Returns 0 when fewer than two models are selected
    or when Ē == 0 (all members perfect).
    """
    idx = np.asarray(subset_indices)
    if len(idx) < 2:
        return 0.0
    if y_true is None:
        raise ValueError("ambiguity diversity requires y_true for normalisation")
    dec = ambiguity_decomposition(predictions_matrix, idx, y_true)
    if dec["avg_member_error"] <= 0.0:
        return 0.0
    return float(np.clip(dec["ambiguity"] / dec["avg_member_error"], 0.0, 1.0))


# ---------------------------------------------------------------------------
# Diversity metric resolution
# ---------------------------------------------------------------------------

#: Valid diversity metrics per task type (all mapped so that higher = more diverse).
CLASSIFICATION_DIVERSITY_METRICS = ("disagreement", "q_statistic", "double_fault")
REGRESSION_DIVERSITY_METRICS = ("ambiguity", "correlation", "rmse_difference")
DEFAULT_DIVERSITY_METRIC = {
    "classification": "disagreement",
    "regression": "ambiguity",
}


def resolve_diversity_metric(diversity_metric: str | None, task_type: str) -> str:
    """Resolve ``'auto'``/None to the task default and validate the combination.

    Raises
    ------
    ValueError
        For an unknown task type or a metric that is invalid for the task.
    """
    if task_type not in DEFAULT_DIVERSITY_METRIC:
        raise ValueError(f"Unknown task_type: {task_type!r}")
    if diversity_metric is None or diversity_metric == "auto":
        return DEFAULT_DIVERSITY_METRIC[task_type]
    valid = (
        CLASSIFICATION_DIVERSITY_METRICS
        if task_type == "classification"
        else REGRESSION_DIVERSITY_METRICS
    )
    if diversity_metric not in valid:
        raise ValueError(
            f"diversity_metric {diversity_metric!r} is not valid for "
            f"task_type {task_type!r}; valid options: {('auto',) + valid}"
        )
    return diversity_metric


# ---------------------------------------------------------------------------
# Pairwise diversity matrix
# ---------------------------------------------------------------------------


def pairwise_diversity_matrix(
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    task_type: str,
    y_true: np.ndarray | None = None,
    diversity_metric: str = "auto",
) -> np.ndarray:
    """Compute pairwise diversity for all model pairs in *subset_indices*.

    All metrics are oriented so that higher values mean *more diverse*:

    * classification ``disagreement`` (default for ``'auto'``): disagreement rate;
    * classification ``q_statistic``: ``(1 − Q) / 2``;
    * classification ``double_fault``: ``1 − DF``;
    * regression ``correlation`` (default for ``'auto'`` here, because
      ambiguity is an ensemble-level, not pairwise, measure): ``1 − |ρ|``;
    * regression ``rmse_difference``: RMS difference between predictions
      (unbounded).

    ``ambiguity`` is not a pairwise measure; requesting it raises ValueError
    (use :func:`average_diversity`).

    Parameters
    ----------
    predictions_matrix : ndarray (n_samples, n_models)
    subset_indices : array-like of int
    task_type : 'classification' | 'regression'
    y_true : ndarray, optional (required for q_statistic / double_fault)
    diversity_metric : str

    Returns
    -------
    ndarray of shape (k, k) where k = len(subset_indices)
        Symmetric matrix with zeros on the diagonal.
    """
    if task_type == "regression" and diversity_metric in (None, "auto"):
        metric = "correlation"
    else:
        metric = resolve_diversity_metric(diversity_metric, task_type)
    if metric == "ambiguity":
        raise ValueError(
            "ambiguity is an ensemble-level measure and has no pairwise matrix; "
            "use average_diversity(..., diversity_metric='ambiguity')"
        )

    idx = np.asarray(subset_indices)
    k = len(idx)

    if k < 2:
        return np.zeros((k, k), dtype=float)

    # Extract subset predictions: (n_samples, k)
    sub_preds = predictions_matrix[:, idx]

    if metric == "disagreement":
        # Vectorized disagreement: (n_samples, k, 1) != (n_samples, 1, k)
        disagree = sub_preds[:, :, np.newaxis] != sub_preds[:, np.newaxis, :]
        matrix = disagree.mean(axis=0).astype(float)
    elif metric in ("q_statistic", "double_fault"):
        if y_true is None:
            raise ValueError(f"diversity_metric {metric!r} requires y_true")
        y = np.asarray(y_true)
        matrix = np.zeros((k, k), dtype=float)
        for a in range(k):
            for b in range(a + 1, k):
                if metric == "q_statistic":
                    val = (1.0 - q_statistic(sub_preds[:, a], sub_preds[:, b], y)) / 2.0
                else:
                    val = 1.0 - double_fault_measure(sub_preds[:, a], sub_preds[:, b], y)
                matrix[a, b] = matrix[b, a] = val
    elif metric == "correlation":
        stds = np.std(sub_preds, axis=0)
        has_zero_std = stds == 0
        if has_zero_std.all():
            # All constant predictions → treat as identical
            matrix = np.zeros((k, k), dtype=float)
        else:
            corr_matrix = np.corrcoef(sub_preds.T)  # (k, k)
            corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)
            corr_matrix[has_zero_std, :] = 1.0
            corr_matrix[:, has_zero_std] = 1.0
            matrix = 1.0 - np.abs(corr_matrix)
    else:  # rmse_difference
        diff = sub_preds[:, :, np.newaxis] - sub_preds[:, np.newaxis, :]
        matrix = np.sqrt(np.mean(diff ** 2, axis=0))

    np.fill_diagonal(matrix, 0.0)
    return matrix


def _mean_pairwise_disagreement(sub_preds: np.ndarray) -> float:
    """Mean pairwise disagreement of k ≥ 2 label columns in O(n·k).

    For one sample with m_c votes for class c, the number of agreeing pairs is
    Σ_c m_c(m_c − 1)/2 out of k(k − 1)/2 pairs, so the mean over pairs and samples
    equals the mean of the upper triangle of the pairwise disagreement matrix
    without building the (n, k, k) array.
    """
    k = sub_preds.shape[1]
    agreeing = np.zeros(sub_preds.shape[0])
    for label in np.unique(sub_preds):
        m = np.count_nonzero(sub_preds == label, axis=1)
        agreeing += m * (m - 1) / 2.0
    return float(1.0 - agreeing.mean() / (k * (k - 1) / 2.0))


def average_diversity(
    predictions_matrix: np.ndarray,
    subset_indices: list[int] | np.ndarray,
    task_type: str,
    y_true: np.ndarray | None = None,
    diversity_metric: str = "auto",
) -> float:
    """Ensemble diversity D(S) (higher → more diverse).

    ``diversity_metric='auto'`` resolves to ``'disagreement'`` for
    classification (mean pairwise disagreement, in [0, 1]) and to
    ``'ambiguity'`` for regression (normalised ambiguity Ā/Ē, in [0, 1]).
    Pairwise metrics are averaged over the upper triangle of
    :func:`pairwise_diversity_matrix`.  Subsets with fewer than two models
    have diversity 0.
    """
    metric = resolve_diversity_metric(diversity_metric, task_type)
    idx = np.asarray(subset_indices)
    k = len(idx)
    if k < 2:
        return 0.0

    if metric == "ambiguity":
        return ambiguity_diversity(predictions_matrix, idx, y_true)
    if metric == "disagreement":
        return _mean_pairwise_disagreement(np.asarray(predictions_matrix)[:, idx])

    matrix = pairwise_diversity_matrix(
        predictions_matrix, idx, task_type, y_true, diversity_metric=metric
    )
    triu_vals = matrix[np.triu_indices(k, k=1)]
    return float(np.mean(triu_vals))
