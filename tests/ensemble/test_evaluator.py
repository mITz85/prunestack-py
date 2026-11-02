"""Tests for prunestack.ensemble.evaluator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prunestack.ensemble.evaluator import EnsembleEvaluator, evaluate_baseline


@pytest.fixture
def classification_setup():
    rng = np.random.RandomState(42)
    n_samples, n_models = 80, 10
    y_true = rng.randint(0, 2, n_samples)
    predictions = np.column_stack([
        (y_true + rng.binomial(1, 0.1 + 0.04 * i, n_samples)) % 2 for i in range(n_models)
    ])
    return y_true, predictions


@pytest.fixture
def regression_setup():
    rng = np.random.RandomState(42)
    n_samples, n_models = 80, 10
    y_true = rng.randn(n_samples) * 10
    predictions = np.column_stack([
        y_true + rng.randn(n_samples) * (0.5 + 0.3 * i) for i in range(n_models)
    ])
    return y_true, predictions


class TestEnsembleEvaluator:
    def test_evaluate_returns_scores(self, classification_setup):
        y, predictions = classification_setup
        result = EnsembleEvaluator(y, predictions, "classification").evaluate(list(range(10)), alpha=0.5)
        assert {"fitness", "performance", "diversity"} <= set(result)
        assert isinstance(result["fitness"], float)

    def test_alpha_grid_shape(self, classification_setup):
        y, predictions = classification_setup
        df = EnsembleEvaluator(y, predictions, "classification").evaluate_alpha_grid(
            list(range(10)), [0.0, 0.5, 1.0],
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert set(df.columns) == {"alpha", "fitness", "performance", "diversity"}

    def test_alpha_grid_default(self, classification_setup):
        y, predictions = classification_setup
        df = EnsembleEvaluator(y, predictions, "classification").evaluate_alpha_grid(list(range(10)))
        assert len(df) == 11

    def test_alpha_grid_only_fitness_varies(self, classification_setup):
        y, predictions = classification_setup
        df = EnsembleEvaluator(y, predictions, "classification").evaluate_alpha_grid([0, 1, 2])
        assert df["performance"].nunique() == 1
        assert df["diversity"].nunique() == 1
        assert df.loc[df["alpha"] == 0.0, "fitness"].iloc[0] == pytest.approx(df["diversity"].iloc[0])

    def test_regression_fitness_in_unit_interval(self, regression_setup):
        y, predictions = regression_setup
        result = EnsembleEvaluator(y, predictions, "regression").evaluate(list(range(10)), 0.5)
        assert 0.0 <= result["fitness"] <= 1.0


class TestEvaluateBaseline:
    def test_keys_and_counts(self, classification_setup):
        y, predictions = classification_setup
        baseline = evaluate_baseline(y, predictions, "classification")
        assert {
            "n_models", "ensemble_performance", "ensemble_diversity", "individual_scores",
            "mean_individual_score", "best_individual_score", "worst_individual_score",
        } <= set(baseline)
        assert baseline["n_models"] == 10
        assert len(baseline["individual_scores"]) == 10

    def test_vote_not_worse_than_worst_model(self, classification_setup):
        y, predictions = classification_setup
        baseline = evaluate_baseline(y, predictions, "classification")
        assert baseline["ensemble_performance"] >= baseline["worst_individual_score"] - 0.01

    def test_regression_reports_raw_rmse(self, regression_setup):
        y, predictions = regression_setup
        baseline = evaluate_baseline(y, predictions, "regression")
        expected = float(np.sqrt(np.mean((predictions.mean(axis=1) - y) ** 2)))
        assert baseline["ensemble_performance"] == pytest.approx(expected)
