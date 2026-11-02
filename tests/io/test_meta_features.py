"""Tests for prunestack.io.meta_features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

from prunestack.io.meta_features import (
    extract_meta_features,
    _extract_dataset_characteristics,
    _extract_base_model_characteristics,
    _extract_ensemble_characteristics,
    _extract_optimization_results,
    extract_optimization_summary,
)

REMOVED_DUPLICATES = {
    "class_entropy", "numerical_feature_ratio", "full_ensemble_diversity", "perf_at_full",
    "perf_spread", "effective_ensemble_size", "error_rate_mean", "error_rate_std", "error_rate_min",
    "model_ensemble_agreement_mean", "model_correlation_mean", "model_correlation_std",
    "model_correlation_max", "model_correlation_min", "model_correlation_median",
}


@pytest.fixture
def classification_setup():
    rng = np.random.RandomState(42)
    n_samples, n_models = 100, 10
    X, y = make_classification(n_samples=n_samples, n_features=15, random_state=42)
    pred_matrix = np.tile(y, (n_models, 1)).T.astype(float)
    noise_mask = rng.random((n_samples, n_models)) < 0.2
    for i in range(n_models):
        pred_matrix[noise_mask[:, i], i] = 1 - pred_matrix[noise_mask[:, i], i]
    return X, y, pred_matrix


@pytest.fixture
def regression_setup():
    rng = np.random.RandomState(42)
    n_samples, n_models = 100, 10
    X, y = make_regression(n_samples=n_samples, n_features=15, random_state=42)
    pred_matrix = np.tile(y, (n_models, 1)).T + rng.normal(0, 10, (n_samples, n_models))
    return X, y, pred_matrix


@pytest.fixture
def sample_optimization_df():
    return pd.DataFrame({
        "alpha": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "best_fitness": [0.6, 0.65, 0.7, 0.72, 0.8, 0.82],
        "n_models_selected": [5, 6, 4, 5, 3, 4],
        "n_iterations": [50, 45, 50, 50, 30, 35],
        "execution_time": [1.0, 1.1, 0.9, 1.0, 0.8, 0.85],
    })


class TestDatasetCharacteristics:

    def test_classification_features(self, classification_setup):
        X, y, _ = classification_setup
        feats = _extract_dataset_characteristics(X, y, "classification")
        assert isinstance(feats, dict)
        assert feats["n_samples"] == 100.0
        assert feats["n_features"] == 15.0
        assert feats["n_classes"] == 2.0
        assert 0 <= feats["class_imbalance_ratio"] <= 1
        assert "mean_feature_correlation" in feats
        assert "label_entropy" in feats

    def test_regression_features(self, regression_setup):
        X, y, _ = regression_setup
        feats = _extract_dataset_characteristics(X, y, "regression")
        assert feats["n_classes"] == 0.0
        assert feats["class_imbalance_ratio"] == 0.0

    def test_same_keys_for_both_tasks(self, classification_setup, regression_setup):
        X_c, y_c, _ = classification_setup
        X_r, y_r, _ = regression_setup
        classification = _extract_dataset_characteristics(X_c, y_c, "classification")
        regression = _extract_dataset_characteristics(X_r, y_r, "regression")
        assert set(classification) == set(regression)

    def test_landmarkers_present(self, classification_setup):
        X, y, _ = classification_setup
        feats = _extract_dataset_characteristics(X, y, "classification")
        landmarkers = {k for k in feats if k.startswith("landmark_")}
        assert landmarkers == {
            f"landmark_{name}_score" for name in (
                "naive_bayes", "decision_stump_depth3", "linear_svm",
                "knn3", "random_forest10", "linear_model",
            )
        }
        assert all(0.0 <= feats[k] <= 1.0 for k in landmarkers)

    def test_no_nan(self, classification_setup):
        X, y, _ = classification_setup
        feats = _extract_dataset_characteristics(X, y, "classification")
        for k, v in feats.items():
            assert not np.isnan(v), f"NaN in feature {k}"


class TestBaseModelCharacteristics:

    def test_classification(self, classification_setup):
        _, y, pred_matrix = classification_setup
        feats = _extract_base_model_characteristics(pred_matrix, y, "classification")
        assert len(feats) == 35
        assert 0 <= feats["model_perf_mean"] <= 1
        assert feats["model_perf_max"] >= feats["model_perf_min"]

    def test_regression(self, regression_setup):
        _, y, pred_matrix = regression_setup
        feats = _extract_base_model_characteristics(pred_matrix, y, "regression")
        assert len(feats) == 35

    def test_no_nan(self, classification_setup):
        _, y, pred_matrix = classification_setup
        feats = _extract_base_model_characteristics(pred_matrix, y, "classification")
        for k, v in feats.items():
            assert not np.isnan(v), f"NaN in feature {k}"


class TestEnsembleCharacteristics:

    def test_classification(self, classification_setup):
        _, y, pred_matrix = classification_setup
        feats = _extract_ensemble_characteristics(pred_matrix, y, "classification")
        assert len(feats) == 40
        assert "diversity_mean" in feats
        assert "full_ensemble_performance" in feats
        assert "oracle_performance" in feats

    def test_no_nan(self, classification_setup):
        _, y, pred_matrix = classification_setup
        feats = _extract_ensemble_characteristics(pred_matrix, y, "classification")
        for k, v in feats.items():
            assert not np.isnan(v), f"NaN in feature {k}"


class TestOptimizationResults:

    def test_with_dataframe(self, sample_optimization_df):
        feats = _extract_optimization_results(sample_optimization_df)
        assert feats["opt_best_fitness_max"] == 0.82
        assert feats["opt_best_alpha"] == 1.0
        assert feats["opt_n_alpha_values"] == 6.0

    def test_none_input_returns_same_keys_as_zero(self, sample_optimization_df):
        feats = _extract_optimization_results(None)
        assert set(feats) == set(_extract_optimization_results(sample_optimization_df))
        assert all(v == 0.0 for v in feats.values())

    def test_no_nan(self, sample_optimization_df):
        feats = _extract_optimization_results(sample_optimization_df)
        for k, v in feats.items():
            assert not np.isnan(v), f"NaN in feature {k}"


class TestExtractMetaFeatures:

    def test_same_keys_for_both_tasks(
        self, classification_setup, regression_setup, sample_optimization_df,
    ):
        keys = []
        for (X, y, predictions), task_type in [
            (classification_setup, "classification"), (regression_setup, "regression"),
        ]:
            feats = extract_meta_features(
                dataset_path="synthetic",
                base_models_df=pd.DataFrame(predictions),
                y_true=y,
                X=X,
                task_type=task_type,
            )
            keys.append(set(feats))
        assert keys[0] == keys[1]
        assert len(keys[0]) == 123

    def test_no_nan_values(self, classification_setup, sample_optimization_df):
        X, y, pred_matrix = classification_setup
        feats = extract_meta_features(
            dataset_path="synthetic",
            base_models_df=pd.DataFrame(pred_matrix),
            y_true=y,
            X=X,
            task_type="classification",
        )
        for k, v in feats.items():
            assert not (isinstance(v, float) and np.isnan(v)), f"NaN in feature {k}"

    def test_correct_types(self, classification_setup, sample_optimization_df):
        X, y, pred_matrix = classification_setup
        feats = extract_meta_features(
            dataset_path="synthetic",
            base_models_df=pd.DataFrame(pred_matrix),
            y_true=y,
            X=X,
            task_type="classification",
        )
        for k, v in feats.items():
            assert isinstance(v, (int, float)), f"Feature {k} has type {type(v)}"

    def test_without_predictions_raises(self, classification_setup):
        X, y, _ = classification_setup
        with pytest.raises(ValueError, match="predictions"):
            extract_meta_features(
                dataset_path="synthetic",
                base_models_df=None,
                y_true=y,
                X=X,
                task_type="classification",
            )


class TestHarmonisationAndLeakage:

    def test_no_optimization_features_in_meta_features(self, classification_setup):
        X, y, pred_matrix = classification_setup
        feats = extract_meta_features("synthetic", None, y, X, "classification", pred_matrix)
        assert not any(k.startswith("opt_") for k in feats)

    def test_optimization_summary_separate(self, sample_optimization_df):
        summary = extract_optimization_summary(sample_optimization_df)
        assert summary["opt_best_alpha"] == 1.0
        assert len(summary) == 24

    def test_removed_duplicates_absent(self, classification_setup):
        X, y, pred_matrix = classification_setup
        feats = extract_meta_features("synthetic", None, y, X, "classification", pred_matrix)
        assert not REMOVED_DUPLICATES & set(feats)

    def test_no_identical_feature_columns_regression(self):
        """Across several datasets no two non-constant meta-features coincide.

        (Regression only: small synthetic classification problems produce
        coincidental equalities, e.g. an oracle accuracy of exactly 1.)
        """
        rows = []
        for seed in range(6):
            rng = np.random.RandomState(seed)
            n, n_models = 80 + 20 * seed, 12
            X, y = make_regression(n_samples=n, n_features=6 + seed % 3 * 4, effective_rank=3 + seed % 2,
                                   noise=5 + seed, random_state=seed)
            scale = (5 + 10 * seed) * rng.uniform(0.5, 2.0, n_models)
            preds = np.tile(y, (n_models, 1)).T + rng.normal(0, 1, (n, n_models)) * scale
            rows.append(extract_meta_features("s", None, y, X, "regression", preds, seed=seed))
        table = pd.DataFrame(rows)
        table = table.loc[:, table.nunique() > 1].round(10)
        duplicated = table.T.duplicated(keep=False)
        assert not duplicated.any(), sorted(table.columns[duplicated.values])

    def test_landmarkers_use_macro_f1(self):
        """On an imbalanced problem macro-F1 based landmarkers are below accuracy."""
        X, y = make_classification(n_samples=300, n_features=8, weights=[0.9, 0.1],
                                   flip_y=0.05, random_state=0)
        feats = _extract_dataset_characteristics(X, y, "classification")
        majority_accuracy = max(np.mean(y == 0), np.mean(y == 1))
        assert feats["landmark_decision_stump_depth3_score"] < majority_accuracy

    def test_regression_scores_are_rmse_skill(self, regression_setup):
        _, y, pred_matrix = regression_setup
        feats = _extract_base_model_characteristics(pred_matrix, y, "regression")
        rmse = np.sqrt(np.mean((pred_matrix[:, 0] - y) ** 2))
        skills = 1 - np.sqrt(np.mean((pred_matrix - y[:, None]) ** 2, axis=0)) / np.std(y)
        assert feats["model_perf_mean"] == pytest.approx(np.mean(np.clip(skills, 0, 1)))
        assert rmse > 0

    def test_landmarkers_use_given_folds(self, classification_setup):
        X, y, _ = classification_setup
        from sklearn.model_selection import StratifiedKFold
        folds = list(StratifiedKFold(5, shuffle=True, random_state=1).split(X, y))
        a = _extract_dataset_characteristics(X, y, "classification", folds=folds)
        b = _extract_dataset_characteristics(X, y, "classification", folds=folds)
        assert a["landmark_knn3_score"] == b["landmark_knn3_score"]
