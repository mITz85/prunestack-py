"""Stacked generalization on top of base-model out-of-fold predictions.

Level-0 inputs
--------------
Regression: the out-of-fold real-valued predictions of the base models.
Classification: by default the out-of-fold **class probabilities**
(``predict_proba``) of the base models, shape ``(n_samples, n_models, K)``, as in
the original multi-response linear regression of Ting & Witten (1999). When no
probabilities are supplied (``stacking_inputs="labels"``), the predicted class
labels ``(n_samples, n_models)`` are used instead. Ensemble pruning and the
diversity measures always use the predicted labels; only the stacking
combiner sees the probabilities.

A level-1 meta-learner is trained on these inputs with an inner k-fold
cross-validation, so every reported stacking score is itself out-of-fold.

Meta-learners
-------------
``"nnls_mlr"`` (classification default)
    Multi-response linear regression with non-negative coefficients
    (Ting & Witten, 1999). For every class ``k`` a non-negative least-squares
    model regresses the indicator ``[y == k]`` on the base-model probabilities
    ``P_m(k | x)`` (or, with label inputs, on the indicators ``[pred_m == k]``);
    the predicted class is the arg-max of the ``K`` responses. It is a learned,
    class-specific weighted (soft) vote.
``"nnls"`` (regression default)
    Non-negative least-squares combination without intercept
    (Breiman, 1996, "Stacked Regressions"): ``y_hat = sum_m w_m * pred_m``, ``w_m >= 0``.
``"ridge"``
    Ridge regression (regression) / ``RidgeClassifier`` (classification).
``"logistic"``
    Multinomial logistic regression (classification only).
``"categorical_nb"``
    Categorical naive Bayes on the predicted labels (classification only;
    always uses label inputs).
``"random_forest"``, ``"hist_gradient_boosting"``
    Non-linear meta-learners (both tasks).

For classification the scikit-learn meta-learners except ``categorical_nb`` see
the probabilities flattened to ``n_models * K`` columns, or the labels one-hot
encoded (``n_models * K`` binary columns) when label inputs are used.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.naive_bayes import CategoricalNB

from prunestack.metrics.performance import calculate_metric, resolve_metric

logger = logging.getLogger(__name__)

DEFAULT_META_LEARNER = {"classification": "nnls_mlr", "regression": "nnls"}
STACKING_INPUTS = ("probabilities", "labels")

_SKLEARN_META_LEARNERS: dict[str, dict[str, type]] = {
    "ridge": {"classification": RidgeClassifier, "regression": Ridge},
    "logistic": {"classification": LogisticRegression},
    "categorical_nb": {"classification": CategoricalNB},
    "random_forest": {"classification": RandomForestClassifier, "regression": RandomForestRegressor},
    "hist_gradient_boosting": {
        "classification": HistGradientBoostingClassifier,
        "regression": HistGradientBoostingRegressor,
    },
}

META_LEARNERS: dict[str, tuple[str, ...]] = {
    "nnls_mlr": ("classification",),
    "nnls": ("regression",),
    **{name: tuple(tasks) for name, tasks in _SKLEARN_META_LEARNERS.items()},
}


def available_meta_learners(task_type: str) -> list[str]:
    """Names of the meta-learners that support ``task_type``."""
    return [name for name, tasks in META_LEARNERS.items() if task_type in tasks]


class _MetaLearner(Protocol):
    def fit(self, predictions: np.ndarray, y: np.ndarray) -> _MetaLearner: ...
    def predict(self, predictions: np.ndarray) -> np.ndarray: ...


def one_hot_labels(predictions: np.ndarray, n_classes: int) -> np.ndarray:
    """Encode an (n_samples, n_models) label matrix as (n_samples, n_models * n_classes)."""
    labels = np.asarray(predictions, dtype=int)
    return np.eye(n_classes, dtype=float)[labels].reshape(labels.shape[0], -1)


class NNLSRegressor:
    """Non-negative linear combination of base-model predictions (no intercept)."""

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> NNLSRegressor:
        weights, _ = nnls(np.asarray(predictions, dtype=float), np.asarray(y, dtype=float))
        if not np.any(weights > 0):
            weights = np.full(predictions.shape[1], 1.0 / predictions.shape[1])
        self.weights_ = weights
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        return np.asarray(predictions, dtype=float) @ self.weights_


class NNLSMultiResponseClassifier:
    """Multi-response linear regression with non-negative weights (Ting & Witten, 1999).

    Accepts either class probabilities ``(n_samples, n_models, K)`` or predicted
    labels ``(n_samples, n_models)``; labels are turned into one-hot indicators,
    so both cases use the same per-class inputs ``Z[:, m, k]``.

    ``weights_[k, m]`` is the weight of base model ``m`` in the response of class ``k``.
    A class whose weights are all zero falls back to equal weights (plain vote).
    Ties between classes are broken by the unweighted sum ``sum_m Z[:, m, k]``
    (soft / plain vote), then by the smallest label.
    """

    def __init__(self, n_classes: int) -> None:
        self.n_classes = n_classes

    def _per_class_inputs(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim == 3:
            return predictions
        labels = predictions.astype(int)
        return np.stack([(labels == k) for k in range(self.n_classes)], axis=2).astype(float)

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> NNLSMultiResponseClassifier:
        z = self._per_class_inputs(predictions)
        y = np.asarray(y, dtype=int)
        n_models = z.shape[1]
        weights = np.zeros((self.n_classes, n_models))
        for k in range(self.n_classes):
            w, _ = nnls(z[:, :, k], (y == k).astype(float))
            weights[k] = w if np.any(w > 0) else 1.0 / n_models
        self.weights_ = weights
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        z = self._per_class_inputs(predictions)
        responses = np.einsum("nmk,km->nk", z, self.weights_)
        votes = z.sum(axis=1)
        is_best = np.isclose(responses, responses.max(axis=1, keepdims=True))
        return np.argmax(np.where(is_best, votes, -1), axis=1)


class _SklearnMetaLearner:
    """Wrap a scikit-learn estimator with the input encoding it needs."""

    def __init__(self, estimator: Any, task_type: str, n_classes: int, one_hot: bool) -> None:
        self.estimator = estimator
        self.task_type = task_type
        self.n_classes = n_classes
        self.one_hot = one_hot

    def _encode(self, predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions)
        if predictions.ndim == 3:  # class probabilities -> n_models * K columns
            return predictions.reshape(predictions.shape[0], -1).astype(float)
        if self.task_type == "classification" and self.one_hot:
            return one_hot_labels(predictions, self.n_classes)
        return np.asarray(predictions, dtype=int if self.task_type == "classification" else float)

    def fit(self, predictions: np.ndarray, y: np.ndarray) -> _SklearnMetaLearner:
        self.estimator_ = clone(self.estimator).fit(self._encode(predictions), y)
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        return self.estimator_.predict(self._encode(predictions))


def build_meta_learner(
    name: str,
    task_type: str,
    n_classes: int = 0,
    params: dict[str, Any] | None = None,
    seed: int = 42,
) -> _MetaLearner:
    """Instantiate a meta-learner by name (see the module docstring)."""
    if name not in META_LEARNERS:
        raise ValueError(f"Unknown meta-learner '{name}'. Available: {sorted(META_LEARNERS)}")
    if task_type not in META_LEARNERS[name]:
        raise ValueError(
            f"Meta-learner '{name}' does not support {task_type}. "
            f"Available: {available_meta_learners(task_type)}"
        )
    params = dict(params or {})

    if name == "nnls_mlr":
        return NNLSMultiResponseClassifier(n_classes=n_classes)
    if name == "nnls":
        return NNLSRegressor()

    estimator_cls = _SKLEARN_META_LEARNERS[name][task_type]
    if name == "categorical_nb":
        params.setdefault("min_categories", n_classes)
    if name == "logistic":
        params.setdefault("max_iter", 1000)
    if "random_state" in estimator_cls().get_params():
        params.setdefault("random_state", seed)
    return _SklearnMetaLearner(
        estimator_cls(**params), task_type, n_classes, one_hot=name != "categorical_nb",
    )


class StackedEnsemble:
    """Level-1 meta-learner trained on base-model out-of-fold predictions.

    Parameters
    ----------
    base_predictions_matrix : ndarray (n_samples, n_models)
        Out-of-fold base-model predictions (encoded class labels 0..K-1 or real values).
    y_true : ndarray (n_samples,)
        Ground truth.
    task_type : 'classification' | 'regression'
    cv_folds : int
        Folds of the inner cross-validation used to score the meta-learner.
    meta_learner : str or None
        Meta-learner name; ``None`` -> ``"nnls_mlr"`` (classification) / ``"nnls"`` (regression).
    meta_learner_params : dict or None
        Keyword arguments for scikit-learn based meta-learners.
    seed : int
        Seed for the inner folds and stochastic meta-learners.
    base_probabilities : ndarray (n_samples, n_models, n_classes) or None
        Out-of-fold class probabilities (classification). When given they are the
        level-0 inputs of every meta-learner except ``categorical_nb``.
    """

    def __init__(
        self,
        base_predictions_matrix: np.ndarray,
        y_true: np.ndarray,
        task_type: str,
        cv_folds: int = 5,
        meta_learner: str | None = None,
        meta_learner_params: dict[str, Any] | None = None,
        seed: int = 42,
        base_probabilities: np.ndarray | None = None,
    ) -> None:
        if task_type not in DEFAULT_META_LEARNER:
            raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}")
        self.base_predictions_matrix = np.asarray(base_predictions_matrix, dtype=float)
        self.y_true = np.asarray(y_true)
        self.task_type = task_type
        self.seed = seed
        self.meta_learner = meta_learner or DEFAULT_META_LEARNER[task_type]
        self.meta_learner_params = dict(meta_learner_params or {})
        self.n_samples, self.n_models = self.base_predictions_matrix.shape

        if task_type == "classification":
            self.y_true = self.y_true.astype(int)
            self.n_classes = int(max(self.y_true.max(), self.base_predictions_matrix.max())) + 1
            class_counts = np.bincount(self.y_true)
            min_class_count = int(class_counts[class_counts > 0].min())
            self.cv_folds = max(2, min(cv_folds, min_class_count))
        else:
            self.n_classes = 0
            self.cv_folds = cv_folds

        # Level-0 inputs: probabilities (n, m, K) or labels / values (n, m).
        self.uses_probabilities = (
            base_probabilities is not None
            and task_type == "classification"
            and self.meta_learner != "categorical_nb"
        )
        if self.uses_probabilities:
            probs = np.asarray(base_probabilities, dtype=float)
            if probs.ndim != 3 or probs.shape[:2] != (self.n_samples, self.n_models):
                raise ValueError(
                    f"base_probabilities must have shape (n_samples, n_models, n_classes) = "
                    f"({self.n_samples}, {self.n_models}, K), got {probs.shape}"
                )
            self.n_classes = max(self.n_classes, probs.shape[2])
            self.level0_inputs = probs
        else:
            self.level0_inputs = self.base_predictions_matrix

        if self.uses_probabilities:
            self.input_kind = "probabilities"
        else:
            self.input_kind = "labels" if task_type == "classification" else "values"

        # Fail fast on an invalid name / task combination.
        self._new_meta_learner()

        self.final_meta_learner: _MetaLearner | None = None
        self.oof_meta_predictions: np.ndarray | None = None
        self.is_trained = False

    def _new_meta_learner(self) -> _MetaLearner:
        return build_meta_learner(
            self.meta_learner, self.task_type, self.n_classes, self.meta_learner_params, self.seed,
        )

    def _folds(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if self.task_type == "classification":
            splitter = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
            return list(splitter.split(np.zeros(self.n_samples), self.y_true))
        splitter = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.seed)
        return list(splitter.split(np.zeros(self.n_samples)))

    def _cross_val_predict(self, predictions: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_samples, dtype=int if self.task_type == "classification" else float)
        for train_idx, val_idx in self._folds():
            learner = self._new_meta_learner().fit(predictions[train_idx], self.y_true[train_idx])
            out[val_idx] = learner.predict(predictions[val_idx])
        return out

    def _subset(self, subset_indices: list[int] | np.ndarray | None) -> np.ndarray:
        if subset_indices is None:
            return self.level0_inputs
        return self.level0_inputs[:, np.asarray(subset_indices, dtype=int)]

    def train(self) -> None:
        """Compute out-of-fold meta-predictions on all base models and fit the final meta-learner."""
        logger.info(
            "Training stacking meta-learner '%s' (%s inputs) with %d-fold CV on %d base models",
            self.meta_learner, "probability" if self.uses_probabilities else "label/value",
            self.cv_folds, self.n_models,
        )
        self.oof_meta_predictions = self._cross_val_predict(self.level0_inputs)
        self.final_meta_learner = self._new_meta_learner().fit(self.level0_inputs, self.y_true)
        self.is_trained = True

    def predict(self, subset_indices: list[int] | np.ndarray | None = None) -> np.ndarray:
        """Out-of-fold meta-predictions using all base models (``None``) or a subset.

        A fresh meta-learner is cross-validated on the selected columns for each subset.
        """
        if subset_indices is None:
            if not self.is_trained:
                raise RuntimeError("Call train() before predict() without a subset")
            return self.oof_meta_predictions.copy()
        return self._cross_val_predict(self._subset(subset_indices))

    def evaluate(
        self,
        subset_indices: list[int] | np.ndarray | None = None,
        performance_metric: str | None = None,
    ) -> float:
        """Raw out-of-fold score of the stacked ensemble (``None`` metric -> macro-F1 / RMSE)."""
        metric = resolve_metric(performance_metric, self.task_type)
        return calculate_metric(self.y_true, self.predict(subset_indices), metric)
