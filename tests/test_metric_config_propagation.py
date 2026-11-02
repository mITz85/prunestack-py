"""Tests verifying that config.performance_metric is respected throughout the pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from prunestack.metrics.performance import (
    calculate_metric,
    calculate_rmse,
    resolve_default_metric,
    resolve_metric,
    is_lower_better,
    evaluate_ensemble_performance,
)
from prunestack.metrics.fitness import calculate_fitness, evaluate_subset
from prunestack.ensemble.stacking import StackedEnsemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regression_data(n=50, seed=42):
    rng = np.random.RandomState(seed)
    y_true = rng.randn(n)
    y_pred = y_true + rng.randn(n) * 0.3
    return y_true, y_pred


def _make_classification_data(n=50, n_classes=3, seed=42):
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, n_classes, size=n)
    y_pred = y_true.copy()
    flip = rng.choice(n, size=n // 10, replace=False)
    y_pred[flip] = (y_pred[flip] + 1) % n_classes
    return y_true, y_pred


# ---------------------------------------------------------------------------
# resolve_default_metric
# ---------------------------------------------------------------------------

class TestResolveDefaultMetric:
    def test_regression_default(self):
        assert resolve_default_metric("regression") == "rmse"

    def test_classification_default(self):
        assert resolve_default_metric("classification") == "f1"


class TestResolveMetric:
    def test_none_returns_default(self):
        assert resolve_metric(None, "regression") == "rmse"
        assert resolve_metric(None, "classification") == "f1"

    def test_empty_returns_default(self):
        assert resolve_metric("", "regression") == "rmse"

    def test_auto_returns_default(self):
        assert resolve_metric("auto", "regression") == "rmse"
        assert resolve_metric("auto", "classification") == "f1"

    def test_explicit_metric_returned(self):
        assert resolve_metric("rmse", "regression") == "rmse"
        assert resolve_metric("f1", "classification") == "f1"

    def test_classification_metric_on_regression_raises(self):
        with pytest.raises(ValueError, match="classification metric"):
            resolve_metric("accuracy", "regression")

    def test_regression_metric_on_classification_raises(self):
        with pytest.raises(ValueError, match="regression metric"):
            resolve_metric("rmse", "classification")


# ---------------------------------------------------------------------------
# calculate_metric dispatching
# ---------------------------------------------------------------------------

class TestCalculateMetric:
    def test_rmse(self):
        y_true, y_pred = _make_regression_data()
        val = calculate_metric(y_true, y_pred, "rmse")
        assert val == pytest.approx(calculate_rmse(y_true, y_pred))
        assert val > 0

    def test_r2(self):
        y_true, y_pred = _make_regression_data()
        from sklearn.metrics import r2_score
        val = calculate_metric(y_true, y_pred, "r2")
        assert val == pytest.approx(r2_score(y_true, y_pred))

    def test_mae(self):
        y_true, y_pred = _make_regression_data()
        from sklearn.metrics import mean_absolute_error
        val = calculate_metric(y_true, y_pred, "mae")
        assert val == pytest.approx(mean_absolute_error(y_true, y_pred))

    def test_accuracy(self):
        y_true, y_pred = _make_classification_data()
        from sklearn.metrics import accuracy_score
        val = calculate_metric(y_true, y_pred, "accuracy")
        assert val == pytest.approx(accuracy_score(y_true, y_pred))


# ---------------------------------------------------------------------------
# is_lower_better
# ---------------------------------------------------------------------------

class TestIsLowerBetter:
    @pytest.mark.parametrize("m", ["rmse", "mae", "mse"])
    def test_error_metrics(self, m):
        assert is_lower_better(m) is True

    @pytest.mark.parametrize("m", ["r2", "accuracy", "f1"])
    def test_score_metrics(self, m):
        assert is_lower_better(m) is False


# ---------------------------------------------------------------------------
# evaluate_ensemble_performance with custom metric
# ---------------------------------------------------------------------------

class TestEvaluateEnsemblePerformanceMetric:
    def test_uses_rmse_when_specified(self):
        y_true, y_pred = _make_regression_data()
        pred_matrix = np.column_stack([y_pred, y_pred + 0.1])
        indices = [0, 1]

        perf_rmse = evaluate_ensemble_performance(
            y_true, pred_matrix, indices, "regression",
            performance_metric="rmse",
        )
        perf_r2 = evaluate_ensemble_performance(
            y_true, pred_matrix, indices, "regression",
            performance_metric="r2",
        )
        # RMSE and R2 should be different values
        assert perf_rmse != pytest.approx(perf_r2, abs=0.01)
        # RMSE should be positive, R2 should be close to 1
        assert perf_rmse > 0
        assert perf_r2 > 0.5


# ---------------------------------------------------------------------------
# calculate_fitness with lower-is-better metric
# ---------------------------------------------------------------------------

class TestFitnessLowerIsBetter:
    def test_evaluate_subset_normalizes_error_metric(self):
        """For error metrics, fitness uses a [0, 1]-normalised skill score.

        The raw error metric (RMSE) is reported as-is in ``performance`` but
        the fitness is computed from a bounded skill score so the α/β
        trade-off stays meaningful (higher fitness ⇒ better, in [0, 1]).
        """
        y_true, y_pred = _make_regression_data()
        pred_matrix = np.column_stack([y_pred, y_pred + 0.1, y_pred - 0.05])
        indices = [0, 1, 2]

        alpha = 0.5
        result = evaluate_subset(
            y_true, pred_matrix, indices,
            alpha=alpha,
            task_type="regression",
            performance_metric="rmse",
        )
        # Raw performance (RMSE) is positive and reported unchanged.
        assert result["performance"] > 0
        # Normalised performance score is bounded in [0, 1] and higher is better.
        assert 0.0 <= result["performance_score"] <= 1.0
        # Fitness is bounded in [0, 1] and consistent with the normalised score.
        assert 0.0 <= result["fitness"] <= 1.0
        expected = (
            alpha * result["performance_score"]
            + (1 - alpha) * result["diversity"]
        )
        assert result["fitness"] == pytest.approx(expected)

    def test_evaluate_subset_no_negation_for_r2(self):
        y_true, y_pred = _make_regression_data()
        pred_matrix = np.column_stack([y_pred, y_pred + 0.1])
        indices = [0, 1]

        result = evaluate_subset(
            y_true, pred_matrix, indices,
            alpha=1.0,
            task_type="regression",
            performance_metric="r2",
        )
        # With alpha=1 (beta=1-alpha=0) and higher-is-better, fitness ≈ performance
        assert result["fitness"] == pytest.approx(result["performance"], abs=0.01)


# ---------------------------------------------------------------------------
# StackedEnsemble.evaluate with custom metric
# ---------------------------------------------------------------------------

class TestStackedEnsembleMetric:
    def test_evaluate_uses_custom_metric(self):
        rng = np.random.RandomState(42)
        n, m = 100, 5
        y_true = rng.randn(n)
        pred_matrix = np.column_stack([y_true + rng.randn(n) * 0.2 for _ in range(m)])

        stacker = StackedEnsemble(
            base_predictions_matrix=pred_matrix,
            y_true=y_true,
            task_type="regression",
            cv_folds=3,
            seed=42,
        )
        stacker.train()

        perf_rmse = stacker.evaluate(performance_metric="rmse")
        perf_r2 = stacker.evaluate(performance_metric="r2")

        # They should give different numeric results
        assert perf_rmse != pytest.approx(perf_r2, abs=0.01)
        assert perf_rmse > 0   # RMSE is positive
        assert perf_r2 > 0.5   # R2 should be high for this easy data


# ---------------------------------------------------------------------------
# End-to-end: train_base_models metric_name in output
# ---------------------------------------------------------------------------

class TestTrainBaseModelsMetricName:
    """``train_base_models`` reports the configured metric in ``metric_name``."""

    @staticmethod
    def _details(dataset_dir, performance_metric):
        from prunestack.data.loader import load_dataset
        from prunestack.models.factory import create_model_pool
        from prunestack.models.hyperparameters import load_hyperparameters
        from prunestack.models.trainer import train_base_models

        bundle = load_dataset(dataset_dir, n_folds=2, seed=42)
        pool = create_model_pool(
            n_features=bundle.n_features,
            hyperparameters=load_hyperparameters(bundle.task_type, seed=42),
            task_type=bundle.task_type,
            max_models_per_algo=1,
        )[:2]
        _, details_df = train_base_models(
            X=bundle.X, y=bundle.y, model_pool=pool, folds=bundle.folds,
            task_type=bundle.task_type, performance_metric=performance_metric,
        )
        return details_df

    def test_explicit_metric(self, classification_dataset_dir):
        details = self._details(classification_dataset_dir, "accuracy")
        assert (details["metric_name"] == "accuracy").all()

    def test_auto_resolves_to_task_default(self, classification_dataset_dir, regression_dataset_dir):
        assert (self._details(classification_dataset_dir, "auto")["metric_name"] == "f1").all()
        assert (self._details(regression_dataset_dir, "auto")["metric_name"] == "rmse").all()
