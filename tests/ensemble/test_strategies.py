"""Tests for prunestack.ensemble.strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prunestack.ensemble.strategies import compare_ensemble_strategies

STRATEGIES = {"full_ensemble", "optimized_ensemble", "full_stacked", "optimized_stacked"}


@pytest.fixture
def classification_setup():
    rng = np.random.RandomState(42)
    n_samples, n_models = 100, 10
    y = rng.randint(0, 2, n_samples)
    predictions = np.tile(y, (n_models, 1)).T.astype(float)
    flip = rng.random((n_samples, n_models)) < 0.2
    predictions[flip] = 1 - predictions[flip]
    optimized_subsets = {0.5: [0, 1, 2, 3, 4], 0.8: [0, 2, 4, 6, 8]}
    return y, predictions, optimized_subsets


@pytest.fixture
def regression_setup():
    rng = np.random.RandomState(42)
    n_samples, n_models = 100, 10
    y = rng.normal(0, 50, n_samples)
    predictions = np.tile(y, (n_models, 1)).T + rng.normal(0, 10, (n_samples, n_models))
    return y, predictions, {0.5: [0, 1, 2]}


class TestCompareEnsembleStrategies:
    def test_columns_and_rows(self, classification_setup):
        y, predictions, subsets = classification_setup
        df = compare_ensemble_strategies(y, predictions, subsets, "classification", cv_folds=3)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["strategy_name", "alpha", "n_models", "performance", "diversity", "fitness"]
        assert set(df["strategy_name"]) == STRATEGIES
        assert len(df) == 4 * len(subsets)
        assert sorted(df["alpha"].unique()) == [0.5, 0.8]

    def test_performance_is_macro_f1(self, classification_setup):
        y, predictions, subsets = classification_setup
        df = compare_ensemble_strategies(y, predictions, subsets, "classification", cv_folds=3)
        assert df["performance"].between(0, 1).all()

    def test_n_models(self, classification_setup):
        y, predictions, subsets = classification_setup
        df = compare_ensemble_strategies(y, predictions, subsets, "classification", cv_folds=3)
        assert (df.loc[df["strategy_name"] == "full_ensemble", "n_models"] == 10).all()
        assert (df.loc[df["strategy_name"] == "optimized_stacked", "n_models"] == 5).all()

    def test_empty_subsets_uses_full_pool(self, classification_setup):
        y, predictions, _ = classification_setup
        df = compare_ensemble_strategies(y, predictions, {}, "classification", cv_folds=3)
        assert len(df) == 4
        assert (df["n_models"] == 10).all()

    @pytest.mark.parametrize("meta_learner", ["logistic", "categorical_nb", "ridge"])
    def test_alternative_meta_learner(self, classification_setup, meta_learner):
        y, predictions, subsets = classification_setup
        df = compare_ensemble_strategies(
            y, predictions, subsets, "classification", cv_folds=3, meta_learner=meta_learner,
        )
        assert set(df["strategy_name"]) == STRATEGIES

    def test_regression(self, regression_setup):
        y, predictions, subsets = regression_setup
        df = compare_ensemble_strategies(y, predictions, subsets, "regression", cv_folds=3)
        assert len(df) == 4
        assert set(df["strategy_name"]) == STRATEGIES
        assert (df["performance"] > 0).all()  # RMSE
