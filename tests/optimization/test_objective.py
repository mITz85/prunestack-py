"""Tests for prunestack.optimization.objective."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.optimization.objective import FitnessObjective


@pytest.fixture
def classification_setup():
    """Simple classification setup for testing."""
    rng = np.random.default_rng(42)
    n_samples, n_models = 50, 5
    y_true = rng.integers(0, 2, size=n_samples)
    # Create predictions that mostly agree with y_true
    predictions = np.tile(y_true, (n_models, 1)).T.astype(float)
    # Add some noise to a few models
    for i in range(n_models):
        flip = rng.choice(n_samples, size=5, replace=False)
        predictions[flip, i] = 1 - predictions[flip, i]
    return y_true, predictions


@pytest.fixture
def regression_setup():
    """Simple regression setup for testing."""
    rng = np.random.default_rng(42)
    n_samples, n_models = 50, 5
    y_true = rng.standard_normal(n_samples)
    predictions = np.tile(y_true, (n_models, 1)).T + rng.standard_normal((n_samples, n_models)) * 0.1
    return y_true, predictions


class TestFitnessObjective:
    """Tests for FitnessObjective wrapper."""

    def test_basic_call(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")
        binary = np.array([1, 1, 0, 0, 0])
        fitness = obj(binary)
        assert isinstance(fitness, float)
        assert fitness > 0

    def test_empty_subset(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")
        binary = np.zeros(5)
        fitness = obj(binary)
        # Scaled fitness may be negative, so the infeasible empty subset is -inf.
        assert fitness == -np.inf

    def test_single_model_subset(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")
        binary = np.array([1, 0, 0, 0, 0])
        fitness = obj(binary)
        assert isinstance(fitness, float)

    def test_full_subset(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")
        binary = np.ones(5)
        fitness = obj(binary)
        assert isinstance(fitness, float)
        assert fitness > 0

    def test_caching(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")
        binary = np.array([1, 1, 0, 0, 0])

        f1 = obj(binary)
        f2 = obj(binary.copy())

        assert f1 == f2
        assert obj.n_calls == 2
        assert obj.n_cache_hits == 1
        assert obj.cache_size == 1

    def test_cache_hit_rate(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")

        obj(np.array([1, 0, 0, 0, 0]))
        obj(np.array([1, 0, 0, 0, 0]))
        obj(np.array([0, 1, 0, 0, 0]))
        obj(np.array([0, 1, 0, 0, 0]))

        assert obj.cache_hit_rate == pytest.approx(0.5)

    def test_minimize_mode(self, classification_setup):
        y_true, preds = classification_setup
        obj_max = FitnessObjective(y_true, preds, alpha=0.5,
                                   task_type="classification", maximize=True)
        obj_min = FitnessObjective(y_true, preds, alpha=0.5,
                                   task_type="classification", maximize=False)

        binary = np.array([1, 1, 1, 0, 0])
        f_max = obj_max(binary)
        f_min = obj_min(binary)

        assert f_max == pytest.approx(-f_min)

    def test_reset_cache(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")
        obj(np.ones(5))
        assert obj.cache_size == 1
        obj.reset_cache()
        assert obj.cache_size == 0
        assert obj.n_calls == 0

    def test_get_best_cached(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")

        assert obj.get_best_cached() is None

        obj(np.array([1, 0, 0, 0, 0]))
        obj(np.ones(5))

        best = obj.get_best_cached()
        assert best is not None
        assert isinstance(best[1], float)

    def test_regression_task(self, regression_setup):
        y_true, preds = regression_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="regression")
        binary = np.array([1, 1, 1, 0, 0])
        fitness = obj(binary)
        assert isinstance(fitness, float)

    def test_repr(self, classification_setup):
        y_true, preds = classification_setup
        obj = FitnessObjective(y_true, preds, alpha=0.5,
                               task_type="classification")
        s = repr(obj)
        assert "FitnessObjective" in s
        assert "alpha=0.5" in s
