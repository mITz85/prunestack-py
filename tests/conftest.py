"""Shared fixtures: small synthetic datasets in the on-disk input format."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TINY_DATASET_DIR = FIXTURES_DIR / "tiny_dataset"


def write_dataset(directory: Path, task_type: str, n_samples: int = 120, n_features: int = 6,
                  n_classes: int = 3, seed: int = 0) -> Path:
    """Write ``X_test.csv``, ``y_test.csv`` and ``metrics.csv`` and return ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    if task_type == "classification":
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=max(2, n_features // 2), n_redundant=0,
                                   n_classes=n_classes, random_state=seed)
        metrics = pd.DataFrame({"accuracy": [0.8], "f1": [0.78]}, index=["DecisionTreeClassifier"])
    else:
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=5.0, random_state=seed)
        metrics = pd.DataFrame({"rmse": [3.1], "r2": [0.9]}, index=["Ridge"])
    pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)]).to_csv(directory / "X_test.csv")
    pd.DataFrame({"target": y}).to_csv(directory / "y_test.csv")
    metrics.to_csv(directory / "metrics.csv")
    return directory


@pytest.fixture
def tiny_dataset_dir() -> Path:
    """Committed 20-sample binary classification dataset."""
    return TINY_DATASET_DIR


@pytest.fixture
def classification_dataset_dir(tmp_path) -> Path:
    """120 samples, 6 features, 3 classes."""
    return write_dataset(tmp_path / "classification", "classification")


@pytest.fixture
def regression_dataset_dir(tmp_path) -> Path:
    """120 samples, 6 features."""
    return write_dataset(tmp_path / "regression", "regression")


@pytest.fixture
def classification_predictions():
    """Labels and an out-of-fold style prediction matrix (100 samples x 8 models)."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=100)
    noise = rng.random((100, 8)) < np.linspace(0.1, 0.5, 8)
    predictions = np.where(noise, rng.integers(0, 3, size=(100, 8)), y[:, None]).astype(float)
    return y, predictions


@pytest.fixture
def regression_predictions():
    """Targets and a prediction matrix (100 samples x 8 models) with increasing noise."""
    rng = np.random.default_rng(0)
    y = rng.normal(10.0, 3.0, size=100)
    predictions = y[:, None] + rng.normal(0.0, np.linspace(0.5, 3.0, 8), size=(100, 8))
    return y, predictions


@pytest.fixture
def classification_metrics_df():
    return pd.DataFrame(
        {"accuracy": [0.75, 0.70], "precision": [0.76, 0.71], "recall": [0.75, 0.70],
         "f1": [0.74, 0.69], "kappa": [0.50, 0.40], "time": [1.0, 2.0]},
        index=["DecisionTreeClassifier", "KNeighborsClassifier"],
    )


@pytest.fixture
def regression_metrics_df():
    return pd.DataFrame(
        {"mae": [2.5, 3.0], "mse": [10.0, 15.0], "rmse": [3.16, 3.87], "r2": [0.95, 0.90], "time": [1.0, 2.0]},
        index=["Ridge", "DecisionTreeRegressor"],
    )
