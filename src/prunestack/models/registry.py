"""Model registry mapping algorithm names to sklearn classes.

Algorithm pool design rationale (for manuscript Methods section)
-----------------------------------------------------------------
Six algorithms were selected per task to maximise *inductive diversity*
across the pool while keeping the end-to-end procedure consistent (every
classifier supports predict_proba and every algorithm scales to all dataset
sizes without special-casing).

Classification (6 algorithms):
  - DecisionTreeClassifier    — axis-aligned recursive partitioning (high
                                 variance, low bias; baseline tree learner)
  - KNeighborsClassifier      — non-parametric, instance-based; decisions
                                 driven purely by local geometry
  - LogisticRegression        — linear probabilistic model; complementary to
                                 non-linear learners, provides well-calibrated
                                 probabilities
  - MLPClassifier             — feedforward neural network; learns non-linear
                                 feature interactions
  - HistGradientBoostingClassifier — gradient boosting on histogram bins;
                                 iterative bias reduction, strong individual
                                 learner
  - RandomForestClassifier    — bagging of trees with internal feature
                                 sampling; diverse from the above via
                                 ensemble-within-a-learner structure and
                                 different bias-variance profile than HGB

Regression (6 algorithms):
  - DecisionTreeRegressor     — same rationale as classification counterpart
  - KNeighborsRegressor       — same rationale as classification counterpart
  - Ridge                     — L2-regularised linear regression; stable
                                 under multicollinearity, low variance
  - MLPRegressor              — same rationale as classification counterpart
  - HistGradientBoostingRegressor — same rationale as classification counterpart
  - ElasticNet                — combined L1+L2 regularisation; sparse solution
                                 finder, complementary to Ridge

All algorithms use fixed hyperparameters (scikit-learn defaults with
iteration-limit adjustments; see configs/hyperparameters.json) that are
identical for the original model and all feature-subset variants. The only
source of within-algorithm variation is the random feature subset (60–90 % of
the features, seed-based).
"""

from __future__ import annotations

import logging

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

logger = logging.getLogger(__name__)

# Base-model pool per task; the order defines the column order of the pool.
_REGISTRY: dict[str, dict[str, type]] = {
    "classification": {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,
        "LogisticRegression": LogisticRegression,
        "MLPClassifier": MLPClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
    },
    "regression": {
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "KNeighborsRegressor": KNeighborsRegressor,
        "Ridge": Ridge,
        "MLPRegressor": MLPRegressor,
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor,
        "ElasticNet": ElasticNet,
    },
}

CLASSIFICATION_ALGORITHMS: list[str] = list(_REGISTRY["classification"])
REGRESSION_ALGORITHMS: list[str] = list(_REGISTRY["regression"])


def get_model_class(algorithm_name: str, task_type: str) -> type:
    """Return the scikit-learn estimator class registered for ``algorithm_name``.

    Raises
    ------
    ValueError
        If ``task_type`` is invalid or the algorithm is not part of that task's pool.
    """
    if task_type not in _REGISTRY:
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got '{task_type}'"
        )
    try:
        return _REGISTRY[task_type][algorithm_name]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}' for {task_type}. "
            f"Available: {list(_REGISTRY[task_type])}"
        ) from None


def list_algorithms(task_type: str | None = None) -> list[str]:
    """Return the registered algorithm names (for one task or for both, sorted)."""
    if task_type is None:
        return sorted(CLASSIFICATION_ALGORITHMS + REGRESSION_ALGORITHMS)
    if task_type not in _REGISTRY:
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got '{task_type}'"
        )
    return list(_REGISTRY[task_type])
