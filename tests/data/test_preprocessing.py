"""Tests for prunestack.data.preprocessing."""

import numpy as np
import pytest

from prunestack.data.preprocessing import create_preprocessing_pipeline


class TestCreatePreprocessingPipeline:
    def test_returns_pipeline(self):
        pipe = create_preprocessing_pipeline()
        assert hasattr(pipe, "fit")
        assert hasattr(pipe, "transform")
        assert hasattr(pipe, "fit_transform")

    def test_has_imputer_and_scaler(self):
        pipe = create_preprocessing_pipeline()
        step_names = [name for name, _ in pipe.steps]
        assert "imputer" in step_names
        assert "scaler" in step_names

    def test_handles_nan(self):
        """Pipeline should impute NaN values."""
        X = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
        pipe = create_preprocessing_pipeline()
        result = pipe.fit_transform(X)
        assert not np.isnan(result).any()

    def test_scales_to_zero_mean_unit_variance(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5) * 10 + 50
        pipe = create_preprocessing_pipeline()
        result = pipe.fit_transform(X)
        # Mean should be ~0, std should be ~1
        np.testing.assert_array_almost_equal(result.mean(axis=0), 0, decimal=10)
        np.testing.assert_array_almost_equal(result.std(axis=0, ddof=0), 1, decimal=10)

    def test_no_leakage_different_stats(self):
        """Fitting on train vs. entire data should give different transforms."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(80, 5) * 10 + 50
        X_val = rng.randn(20, 5) * 10 + 100  # Different distribution

        # Fit on train only
        pipe_train = create_preprocessing_pipeline()
        pipe_train.fit(X_train)
        val_from_train = pipe_train.transform(X_val)

        # Fit on all data (leaky)
        pipe_all = create_preprocessing_pipeline()
        X_all = np.vstack([X_train, X_val])
        pipe_all.fit(X_all)
        val_from_all = pipe_all.transform(X_val)

        # Results should differ (different scaling stats)
        assert not np.allclose(val_from_train, val_from_all)

    def test_transform_preserves_shape(self):
        X = np.random.randn(50, 10)
        pipe = create_preprocessing_pipeline()
        result = pipe.fit_transform(X)
        assert result.shape == (50, 10)

    def test_each_call_returns_new_pipeline(self):
        """Each call should return a fresh unfitted pipeline."""
        p1 = create_preprocessing_pipeline()
        p2 = create_preprocessing_pipeline()
        assert p1 is not p2
