"""Tests for prunestack.models.registry."""

import pytest
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from prunestack.models.registry import get_model_class, list_algorithms

CLASSIFICATION = [
    DecisionTreeClassifier, KNeighborsClassifier, LogisticRegression,
    MLPClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
]
REGRESSION = [
    DecisionTreeRegressor, KNeighborsRegressor, Ridge,
    MLPRegressor, HistGradientBoostingRegressor, ElasticNet,
]


class TestGetModelClass:
    @pytest.mark.parametrize(
        "task_type, estimator",
        [("classification", c) for c in CLASSIFICATION] + [("regression", r) for r in REGRESSION],
    )
    def test_returns_registered_class(self, task_type, estimator):
        assert get_model_class(estimator.__name__, task_type) is estimator

    @pytest.mark.parametrize(
        "name, task_type",
        [("DecisionTreeRegressor", "classification"), ("LogisticRegression", "regression"),
         ("LinearSVC", "classification"), ("UnknownModel", "regression")],
    )
    def test_algorithm_outside_task_pool_raises(self, name, task_type):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_model_class(name, task_type)

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError, match="task_type must be"):
            get_model_class("DecisionTreeClassifier", "clustering")


class TestListAlgorithms:
    def test_all_algorithms_sorted(self):
        algorithms = list_algorithms()
        assert algorithms == sorted(c.__name__ for c in CLASSIFICATION + REGRESSION)

    def test_per_task(self):
        assert set(list_algorithms("classification")) == {c.__name__ for c in CLASSIFICATION}
        assert set(list_algorithms("regression")) == {r.__name__ for r in REGRESSION}

    def test_invalid_task_type_raises(self):
        with pytest.raises(ValueError):
            list_algorithms("clustering")
