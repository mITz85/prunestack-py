"""Tests for prunestack.metrics.diversity."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.metrics.diversity import (
    q_statistic,
    disagreement_measure,
    double_fault_measure,
    correlation_coefficient,
    rmse_difference,
    pairwise_diversity_matrix,
    average_diversity,
)


# ---------------------------------------------------------------------------
# Classification pairwise metrics
# ---------------------------------------------------------------------------

class TestQStatistic:
    def test_identical_classifiers(self):
        y = np.array([0, 1, 1, 0, 1])
        p = np.array([0, 1, 0, 0, 1])  # same model
        q = q_statistic(p, p, y)
        # When identical, N01=N10=0, so Q=1 (or 0 if denominator is 0)
        # N11*N00 > 0 and N01*N10 = 0, so Q = 1.0
        assert q == pytest.approx(1.0)

    def test_opposite_classifiers(self):
        y = np.array([0, 1, 1, 0, 1])
        # p1 mostly correct, p2 mostly wrong but with some overlap
        p1 = np.array([0, 1, 0, 0, 1])  # 4/5 correct
        p2 = np.array([1, 0, 1, 0, 0])  # 1/5 correct
        q = q_statistic(p1, p2, y)
        # Should be negative (anti-correlated)
        assert q < 0.0

    def test_all_wrong_both(self):
        # Both perfectly wrong → N00=5, others=0 → denom=0 → 0.0
        y = np.array([0, 1, 1, 0, 1])
        p1 = np.array([1, 0, 0, 1, 0])
        p2 = np.array([1, 0, 0, 1, 0])
        q = q_statistic(p1, p2, y)
        assert q == 0.0  # degenerate case

    def test_q_range(self):
        rng = np.random.RandomState(42)
        y = rng.randint(0, 2, 100)
        p1 = rng.randint(0, 2, 100)
        p2 = rng.randint(0, 2, 100)
        q = q_statistic(p1, p2, y)
        assert -1.0 <= q <= 1.0

    def test_zero_denominator(self):
        # All correct, all agree
        y = np.array([1, 1, 1])
        p1 = np.array([1, 1, 1])
        p2 = np.array([1, 1, 1])
        q = q_statistic(p1, p2, y)
        # N11=3, N00=0, N01=0, N10=0 → denom = 0 → returns 0
        assert q == 0.0


class TestDisagreementMeasure:
    def test_identical(self):
        p = np.array([0, 1, 1, 0])
        assert disagreement_measure(p, p) == 0.0

    def test_opposite(self):
        p1 = np.array([0, 0, 0, 0])
        p2 = np.array([1, 1, 1, 1])
        assert disagreement_measure(p1, p2) == 1.0

    def test_partial(self):
        p1 = np.array([0, 1, 0, 1])
        p2 = np.array([0, 0, 0, 1])
        assert disagreement_measure(p1, p2) == pytest.approx(0.25)


class TestDoubleFaultMeasure:
    def test_no_double_fault(self):
        y = np.array([0, 1, 0, 1])
        p1 = np.array([0, 1, 0, 1])  # perfect
        p2 = np.array([0, 1, 0, 1])  # perfect
        assert double_fault_measure(p1, p2, y) == 0.0

    def test_all_double_fault(self):
        y = np.array([0, 1, 0, 1])
        p1 = np.array([1, 0, 1, 0])  # all wrong
        p2 = np.array([1, 0, 1, 0])  # all wrong
        assert double_fault_measure(p1, p2, y) == 1.0

    def test_partial_double_fault(self):
        y = np.array([0, 1, 0, 1])
        p1 = np.array([1, 1, 0, 1])  # 1 wrong
        p2 = np.array([1, 0, 0, 1])  # 1 wrong (same sample)
        df = double_fault_measure(p1, p2, y)
        assert df == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Regression pairwise metrics
# ---------------------------------------------------------------------------

class TestCorrelationCoefficient:
    def test_perfect_correlation(self):
        p1 = np.array([1.0, 2.0, 3.0, 4.0])
        p2 = np.array([2.0, 4.0, 6.0, 8.0])
        assert correlation_coefficient(p1, p2) == pytest.approx(1.0)

    def test_negative_correlation(self):
        p1 = np.array([1.0, 2.0, 3.0, 4.0])
        p2 = np.array([4.0, 3.0, 2.0, 1.0])
        assert correlation_coefficient(p1, p2) == pytest.approx(-1.0)

    def test_constant_predictions(self):
        p1 = np.array([5.0, 5.0, 5.0])
        p2 = np.array([1.0, 2.0, 3.0])
        # constant → treated as identical (returns 1.0)
        assert correlation_coefficient(p1, p2) == 1.0

    def test_uncorrelated(self):
        rng = np.random.RandomState(42)
        p1 = rng.randn(1000)
        p2 = rng.randn(1000)
        cc = correlation_coefficient(p1, p2)
        assert abs(cc) < 0.1  # roughly uncorrelated


class TestRmseDifference:
    def test_identical(self):
        p = np.array([1.0, 2.0, 3.0])
        assert rmse_difference(p, p) == pytest.approx(0.0)

    def test_known_value(self):
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([2.0, 3.0, 4.0])
        assert rmse_difference(p1, p2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Pairwise matrix and average
# ---------------------------------------------------------------------------

class TestPairwiseDiversityMatrix:
    def test_classification_shape(self):
        pred = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ])
        mat = pairwise_diversity_matrix(pred, [0, 1, 2], "classification")
        assert mat.shape == (3, 3)

    def test_symmetric(self):
        rng = np.random.RandomState(42)
        pred = rng.randint(0, 2, (50, 5))
        mat = pairwise_diversity_matrix(pred, [0, 1, 2, 3, 4], "classification")
        np.testing.assert_array_almost_equal(mat, mat.T)

    def test_diagonal_zero(self):
        rng = np.random.RandomState(42)
        pred = rng.randint(0, 2, (50, 4))
        mat = pairwise_diversity_matrix(pred, [0, 1, 2, 3], "classification")
        np.testing.assert_array_almost_equal(np.diag(mat), 0.0)

    def test_identical_models_zero_diversity(self):
        # All columns identical
        col = np.array([0, 1, 0, 1, 0])
        pred = np.column_stack([col, col, col])
        mat = pairwise_diversity_matrix(pred, [0, 1, 2], "classification")
        np.testing.assert_array_almost_equal(mat, 0.0)

    def test_regression(self):
        rng = np.random.RandomState(42)
        pred = rng.randn(50, 4)
        mat = pairwise_diversity_matrix(pred, [0, 1, 2, 3], "regression")
        assert mat.shape == (4, 4)
        np.testing.assert_array_almost_equal(np.diag(mat), 0.0)

    def test_unknown_task_type(self):
        with pytest.raises(ValueError):
            pairwise_diversity_matrix(np.zeros((3, 2)), [0, 1], "unknown")


class TestAverageDiversity:
    def test_single_model(self):
        pred = np.array([[0], [1], [0]])
        assert average_diversity(pred, [0], "classification") == 0.0

    def test_identical_models(self):
        col = np.array([0, 1, 0, 1])
        pred = np.column_stack([col, col, col])
        assert average_diversity(pred, [0, 1, 2], "classification") == pytest.approx(0.0)

    def test_opposite_models(self):
        p1 = np.array([0, 0, 0, 0])
        p2 = np.array([1, 1, 1, 1])
        pred = np.column_stack([p1, p2])
        div = average_diversity(pred, [0, 1], "classification")
        assert div == pytest.approx(1.0)

    def test_range(self):
        rng = np.random.RandomState(42)
        pred = rng.randint(0, 2, (100, 10))
        div = average_diversity(pred, list(range(10)), "classification")
        assert 0.0 <= div <= 1.0

    def test_regression_range(self):
        rng = np.random.RandomState(42)
        pred = rng.randn(100, 10)
        y = rng.randn(100)
        # Default ('auto' → ambiguity) needs y_true for normalisation
        div = average_diversity(pred, list(range(10)), "regression", y)
        assert 0.0 <= div <= 1.0
        div_corr = average_diversity(
            pred, list(range(10)), "regression", diversity_metric="correlation"
        )
        assert 0.0 <= div_corr <= 1.0


def test_vote_count_disagreement_equals_pairwise_mean():
    from itertools import combinations

    from prunestack.metrics.diversity import _mean_pairwise_disagreement
    rng = np.random.default_rng(3)
    preds = rng.integers(0, 4, size=(200, 7)).astype(float)
    pairwise = np.mean([np.mean(preds[:, i] != preds[:, j]) for i, j in combinations(range(7), 2)])
    assert _mean_pairwise_disagreement(preds) == pytest.approx(pairwise)
