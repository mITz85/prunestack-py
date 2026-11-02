"""Tests for prunestack.optimization.runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prunestack.optimization.eho import EHOOptimizer
import json

from prunestack.optimization.runner import optimize_alpha_grid, optimize_ensemble
from prunestack.optimization.protocol import OptimizationResult


@pytest.fixture
def classification_data():
    """Small classification dataset for testing."""
    rng = np.random.default_rng(42)
    n_samples, n_models = 50, 8
    y_true = rng.integers(0, 2, size=n_samples)
    predictions = np.tile(y_true, (n_models, 1)).T.astype(float)
    for i in range(n_models):
        flip = rng.choice(n_samples, size=max(3, n_samples // 10), replace=False)
        predictions[flip, i] = 1 - predictions[flip, i]
    return y_true, predictions


@pytest.fixture
def small_optimizer():
    """Small/fast optimizer for testing."""
    return EHOOptimizer(
        n_clans=2, clan_size=3, max_iterations=5,
        early_stop_patience=0,
    )


class TestOptimizeEnsemble:
    """Tests for single optimization run."""

    def test_basic_run(self, classification_data, small_optimizer):
        y_true, preds = classification_data
        result = optimize_ensemble(
            y_true=y_true,
            predictions_matrix=preds,
            alpha=0.5,
            task_type="classification",
            optimizer=small_optimizer,
            seed=42,
        )
        assert isinstance(result, OptimizationResult)
        assert result.best_fitness > 0
        assert result.n_selected >= 1
        assert result.n_selected <= preds.shape[1]

    def test_metadata_has_objective_stats(self, classification_data, small_optimizer):
        y_true, preds = classification_data
        result = optimize_ensemble(
            y_true=y_true,
            predictions_matrix=preds,
            alpha=0.5,
            task_type="classification",
            optimizer=small_optimizer,
            seed=42,
        )
        assert "objective_calls" in result.metadata
        assert "cache_hits" in result.metadata
        assert result.metadata["objective_calls"] > 0

    def test_determinism(self, classification_data, small_optimizer):
        y_true, preds = classification_data
        r1 = optimize_ensemble(y_true, preds, 0.5, "classification",
                               small_optimizer, seed=42)
        r2 = optimize_ensemble(y_true, preds, 0.5, "classification",
                               small_optimizer, seed=42)
        assert r1.best_fitness == r2.best_fitness
        np.testing.assert_array_equal(r1.best_subset, r2.best_subset)


class TestOptimizeAlphaGrid:
    def test_one_row_per_alpha(self, classification_data, small_optimizer):
        y_true, preds = classification_data
        df = optimize_alpha_grid(y_true, preds, [0.0, 0.5, 1.0], optimizer=small_optimizer,
                                 task_type="classification", seed=42)
        assert list(df["alpha"]) == [0.0, 0.5, 1.0]
        assert list(df.columns) == ["alpha", "best_fitness", "best_subset_indices", "n_models_selected",
                                    "performance", "performance_score", "diversity",
                                    "n_evaluations", "n_iterations", "execution_time", "convergence_history"]
        assert "beta" not in df.columns

    def test_json_columns_roundtrip(self, classification_data, small_optimizer):
        y_true, preds = classification_data
        df = optimize_alpha_grid(y_true, preds, [0.5], optimizer=small_optimizer,
                                 task_type="classification", seed=42)
        indices = json.loads(df.loc[0, "best_subset_indices"])
        history = json.loads(df.loc[0, "convergence_history"])
        assert len(indices) == df.loc[0, "n_models_selected"]
        assert all(0 <= i < preds.shape[1] for i in indices)
        assert history == sorted(history)  # best fitness never decreases

    def test_deterministic(self, classification_data, small_optimizer):
        y_true, preds = classification_data
        kwargs = dict(optimizer=small_optimizer, task_type="classification", seed=3)
        a = optimize_alpha_grid(y_true, preds, [0.0, 1.0], **kwargs)
        b = optimize_alpha_grid(y_true, preds, [0.0, 1.0], **kwargs)
        pd.testing.assert_frame_equal(a.drop(columns="execution_time"), b.drop(columns="execution_time"))

    def test_fitness_non_negative(self, classification_data, small_optimizer):
        y_true, preds = classification_data
        df = optimize_alpha_grid(y_true, preds, [0.0, 0.5, 1.0], optimizer=small_optimizer,
                                 task_type="classification", seed=42)
        assert (df["best_fitness"] >= 0).all()
