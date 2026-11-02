"""Tests for prunestack.metrics.performance."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.metrics.performance import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_mae,
    calculate_mse,
    calculate_r2,
    ensemble_predictions_classification,
    ensemble_predictions_regression,
    evaluate_ensemble_performance,
)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

class TestClassificationMetrics:
    def test_accuracy_perfect(self):
        y = np.array([0, 1, 1, 0, 1])
        assert calculate_accuracy(y, y) == 1.0

    def test_accuracy_zero(self):
        y = np.array([0, 0, 0])
        p = np.array([1, 1, 1])
        assert calculate_accuracy(y, p) == 0.0

    def test_accuracy_partial(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0, 0, 0, 1])
        assert calculate_accuracy(y, p) == pytest.approx(0.75)

    def test_f1_perfect(self):
        y = np.array([0, 1, 1, 0])
        assert calculate_f1_score(y, y) == pytest.approx(1.0)

    def test_f1_weighted(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0, 0, 1, 0])
        f1 = calculate_f1_score(y, p, average="weighted")
        assert 0.0 <= f1 <= 1.0

    def test_f1_macro(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([0, 0, 1, 0])
        f1 = calculate_f1_score(y, p, average="macro")
        assert 0.0 <= f1 <= 1.0


class TestEnsemblePredictionsClassification:
    def test_majority_vote_unanimous(self):
        # All 3 models agree
        pred = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ])
        result = ensemble_predictions_classification(pred, [0, 1, 2])
        np.testing.assert_array_equal(result, [0, 1, 0])

    def test_majority_vote_disagreement(self):
        # 2 vs 1
        pred = np.array([
            [0, 1, 0],
            [1, 0, 1],
        ])
        result = ensemble_predictions_classification(pred, [0, 1, 2])
        np.testing.assert_array_equal(result, [0, 1])

    def test_majority_vote_subset(self):
        pred = np.array([
            [0, 1, 0, 1],
            [1, 0, 0, 1],
        ])
        # Only models 0 and 3
        result = ensemble_predictions_classification(pred, [0, 3])
        # Row 0: [0,1] → tie → mode picks lower; Row 1: [1,1] → 1
        assert result[1] == 1


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestRegressionMetrics:
    def test_mae(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.5, 2.5, 3.5])
        assert calculate_mae(y, p) == pytest.approx(0.5)

    def test_mse(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.0, 2.0, 4.0])
        assert calculate_mse(y, p) == pytest.approx(1.0 / 3.0)

    def test_r2_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert calculate_r2(y, y) == pytest.approx(1.0)

    def test_r2_range(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = np.array([1.2, 1.8, 3.1, 4.3, 4.7])
        r2 = calculate_r2(y, p)
        assert 0.0 < r2 < 1.0


class TestEnsemblePredictionsRegression:
    def test_mean_aggregation(self):
        pred = np.array([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ])
        result = ensemble_predictions_regression(pred, [0, 1, 2])
        np.testing.assert_array_almost_equal(result, [3.0, 4.0])

    def test_subset(self):
        pred = np.array([
            [1.0, 3.0, 5.0],
            [2.0, 4.0, 6.0],
        ])
        result = ensemble_predictions_regression(pred, [0, 2])
        np.testing.assert_array_almost_equal(result, [3.0, 4.0])


# ---------------------------------------------------------------------------
# Unified evaluator
# ---------------------------------------------------------------------------

class TestEvaluateEnsemblePerformance:
    def test_classification(self):
        y = np.array([0, 1, 1, 0, 1])
        pred = np.array([
            [0, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1],
        ])
        perf = evaluate_ensemble_performance(y, pred, [0, 1, 2], "classification")
        assert 0.0 <= perf <= 1.0

    def test_regression(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([
            [1.1, 0.9],
            [2.1, 1.9],
            [3.1, 2.9],
        ])
        perf = evaluate_ensemble_performance(y, pred, [0, 1], "regression")
        assert perf == pytest.approx(0.0)  # mean of the two models is exact -> RMSE 0

    def test_regression_r2_is_not_clipped(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([[100.0], [100.0], [100.0]])
        perf = evaluate_ensemble_performance(y, pred, [0], "regression", performance_metric="r2")
        assert perf < 0.0

    def test_unknown_task_type(self):
        with pytest.raises(ValueError, match="Unknown task_type"):
            evaluate_ensemble_performance(
                np.array([1]), np.array([[1]]), [0], "invalid"
            )
