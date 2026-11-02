"""Tests for prunestack.models.trainer."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold

from prunestack.models.factory import create_model_pool, ModelConfig
from prunestack.models.trainer import train_base_models


@pytest.fixture
def classification_data():
    """Small synthetic classification dataset."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(50, 8), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng.choice([0, 1, 2], size=50))
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(cv.split(X, y))
    return X, y, folds


@pytest.fixture
def regression_data():
    """Small synthetic regression dataset."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(50, 8), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(rng.randn(50) * 10 + 5)
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(cv.split(X))
    return X, y, folds


@pytest.fixture
def classification_hyperparameters():
    return {
        "DecisionTreeClassifier": {"max_depth": 3, "random_state": 42},
        "KNeighborsClassifier": {"n_neighbors": 3},
    }


@pytest.fixture
def regression_hyperparameters():
    return {
        "DecisionTreeRegressor": {"max_depth": 3, "random_state": 42},
        "KNeighborsRegressor": {"n_neighbors": 3},
    }


class TestTrainBaseModels:
    def test_predictions_shape_classification(self, classification_data, classification_hyperparameters):
        X, y, folds = classification_data
        pool = create_model_pool(
            n_features=8, hyperparameters=classification_hyperparameters,
            task_type="classification", max_models_per_algo=3,
        )
        preds, details = train_base_models(X, y, pool, folds, "classification")
        # 2 algos * 3 models = 6 models
        assert preds.shape == (50, 6)

    def test_predictions_shape_regression(self, regression_data, regression_hyperparameters):
        X, y, folds = regression_data
        pool = create_model_pool(
            n_features=8, hyperparameters=regression_hyperparameters,
            task_type="regression", max_models_per_algo=3,
        )
        preds, details = train_base_models(X, y, pool, folds, "regression")
        assert preds.shape == (50, 6)

    def test_no_nan_in_predictions(self, classification_data, classification_hyperparameters):
        X, y, folds = classification_data
        pool = create_model_pool(
            n_features=8, hyperparameters=classification_hyperparameters,
            task_type="classification", max_models_per_algo=3,
        )
        preds, _ = train_base_models(X, y, pool, folds, "classification")
        assert preds.isna().sum().sum() == 0

    def test_out_of_fold_predictions(self, classification_data, classification_hyperparameters):
        """Each sample should be predicted exactly once (when it's in validation)."""
        X, y, folds = classification_data
        pool = create_model_pool(
            n_features=8, hyperparameters=classification_hyperparameters,
            task_type="classification", max_models_per_algo=2,
        )
        preds, _ = train_base_models(X, y, pool, folds, "classification")
        # All values should be filled (no NaN = each sample covered once)
        assert preds.isna().sum().sum() == 0
        # Each column should have valid predictions for all 50 samples
        assert preds.shape[0] == 50

    def test_details_df_structure(self, classification_data, classification_hyperparameters):
        X, y, folds = classification_data
        pool = create_model_pool(
            n_features=8, hyperparameters=classification_hyperparameters,
            task_type="classification", max_models_per_algo=2,
        )
        _, details = train_base_models(X, y, pool, folds, "classification")
        expected_cols = {
            "model_id", "model_name", "algorithm", "fold",
            "n_features_used", "is_original", "metric_name",
            "metric_value", "train_samples", "val_samples", "fold_time_sec",
        }
        assert set(details.columns) == expected_cols
        # 4 models * 3 folds = 12 rows
        assert len(details) == 4 * 3

    def test_deterministic(self, classification_data, classification_hyperparameters):
        X, y, folds = classification_data
        pool = create_model_pool(
            n_features=8, hyperparameters=classification_hyperparameters,
            task_type="classification", max_models_per_algo=2,
        )
        preds1, _ = train_base_models(X, y, pool, folds, "classification")
        preds2, _ = train_base_models(X, y, pool, folds, "classification")
        pd.testing.assert_frame_equal(preds1, preds2)

    def test_column_names(self, classification_data, classification_hyperparameters):
        X, y, folds = classification_data
        pool = create_model_pool(
            n_features=8, hyperparameters=classification_hyperparameters,
            task_type="classification", max_models_per_algo=2,
        )
        preds, _ = train_base_models(X, y, pool, folds, "classification")
        assert list(preds.columns) == [mc.name for mc in pool]
        assert preds.columns[0] == "DecisionTreeClassifier_original"
        assert preds.columns[1] == "DecisionTreeClassifier_subset_01"


class TestFoldMetric:
    def test_default_metric_is_macro_f1(self, classification_data, classification_hyperparameters):
        X, y, folds = classification_data
        pool = create_model_pool(n_features=8, hyperparameters=classification_hyperparameters,
                                 task_type="classification", max_models_per_algo=1)
        _, details = train_base_models(X, y, pool, folds, "classification")
        assert set(details["metric_name"]) == {"f1"}

    def test_default_metric_is_rmse(self, regression_data, regression_hyperparameters):
        X, y, folds = regression_data
        pool = create_model_pool(n_features=8, hyperparameters=regression_hyperparameters,
                                 task_type="regression", max_models_per_algo=1)
        _, details = train_base_models(X, y, pool, folds, "regression")
        assert set(details["metric_name"]) == {"rmse"}
        assert (details["metric_value"] >= 0).all()


def test_return_probabilities_classification():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold
    from prunestack.models.factory import create_model_pool
    from prunestack.models.hyperparameters import load_hyperparameters
    X, y = make_classification(n_samples=120, n_features=6, n_informative=4, n_classes=3, random_state=0)
    pool = create_model_pool(6, load_hyperparameters("classification", 0), "classification",
                             base_seed=0, max_models_per_algo=1)
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(X, y))
    preds, _, proba = train_base_models(X, y, pool, folds, "classification", return_probabilities=True)
    assert proba.shape == (120, len(pool), 3)
    assert np.allclose(proba.sum(axis=2), 1.0)
    # Hard predictions agree with the arg-max of the probabilities for most samples.
    assert np.mean(proba.argmax(axis=2) == preds.to_numpy()) > 0.9


def test_return_probabilities_regression_is_none():
    from sklearn.datasets import make_regression
    from sklearn.model_selection import KFold
    from prunestack.models.factory import create_model_pool
    from prunestack.models.hyperparameters import load_hyperparameters
    X, y = make_regression(n_samples=60, n_features=4, random_state=0)
    pool = create_model_pool(4, load_hyperparameters("regression", 0), "regression",
                             base_seed=0, max_models_per_algo=1)
    folds = list(KFold(3, shuffle=True, random_state=0).split(X))
    _, _, proba = train_base_models(X, y, pool, folds, "regression", return_probabilities=True)
    assert proba is None


class TestParallelTraining:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_n_jobs_gives_identical_results(self, task, classification_data, regression_data,
                                            classification_hyperparameters, regression_hyperparameters):
        X, y, folds = classification_data if task == "classification" else regression_data
        hp = classification_hyperparameters if task == "classification" else regression_hyperparameters
        pool = create_model_pool(n_features=8, hyperparameters=hp, task_type=task, max_models_per_algo=3)
        serial = train_base_models(X, y, pool, folds, task, return_probabilities=True, n_jobs=1)
        parallel = train_base_models(X, y, pool, folds, task, return_probabilities=True, n_jobs=2)
        pd.testing.assert_frame_equal(serial[0], parallel[0])
        cols = ["model_id", "fold", "metric_value"]
        pd.testing.assert_frame_equal(serial[1][cols], parallel[1][cols])
        if task == "classification":
            np.testing.assert_array_equal(serial[2], parallel[2])
        else:
            assert serial[2] is None and parallel[2] is None
