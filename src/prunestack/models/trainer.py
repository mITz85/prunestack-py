"""Base model trainer with cross-validation and no-leakage preprocessing."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[assignment]

from prunestack.data.preprocessing import create_preprocessing_pipeline
from prunestack.metrics.performance import calculate_metric, resolve_metric
from prunestack.models.factory import ModelConfig
from prunestack.models.registry import get_model_class

logger = logging.getLogger(__name__)


def _fit_model_cv(
    model_cfg: ModelConfig,
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    task_type: str,
    performance_metric: str,
    n_classes: int | None,
) -> tuple[np.ndarray, np.ndarray | None, list[dict[str, Any]]]:
    """k-fold CV of one model: out-of-fold predictions, probabilities and per-fold records.

    For each fold the features of the model are selected, the preprocessing pipeline is
    fitted on the training part ONLY (no leakage), the model is trained and the
    validation part is predicted. ``n_classes`` is ``None`` when no probabilities are
    collected.
    """
    model_start = time.time()
    feat_idx = model_cfg.feature_indices
    predictions = np.full(X_arr.shape[0], np.nan)
    probabilities = np.full((X_arr.shape[0], n_classes), np.nan) if n_classes is not None else None
    records: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_start = time.time()
        X_train_fold = X_arr[np.ix_(train_idx, feat_idx)]
        X_val_fold = X_arr[np.ix_(val_idx, feat_idx)]

        pipeline = create_preprocessing_pipeline()
        X_train_processed = pipeline.fit_transform(X_train_fold)
        X_val_processed = pipeline.transform(X_val_fold)

        model = get_model_class(model_cfg.algorithm, task_type)(**model_cfg.params)
        model.fit(X_train_processed, y_arr[train_idx])

        y_pred = model.predict(X_val_processed)
        predictions[val_idx] = y_pred
        if probabilities is not None:
            # Map the model's classes_ onto the global class columns 0..K-1
            # (a class absent from a training fold keeps probability 0).
            fold_proba = np.zeros((len(val_idx), n_classes))
            fold_proba[:, np.asarray(model.classes_, dtype=int)] = model.predict_proba(X_val_processed)
            probabilities[val_idx, :] = fold_proba

        records.append({
            "model_id": model_cfg.model_id,
            "model_name": model_cfg.name,
            "algorithm": model_cfg.algorithm,
            "fold": fold_idx,
            "n_features_used": model_cfg.n_features_used,
            "is_original": model_cfg.is_original,
            "metric_name": performance_metric,
            "metric_value": calculate_metric(y_arr[val_idx], y_pred, performance_metric),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "fold_time_sec": round(time.time() - fold_start, 4),
        })

    logger.debug("Model %s (%d features) completed in %.2fs",
                 model_cfg.name, model_cfg.n_features_used, time.time() - model_start)
    return predictions, probabilities, records


def train_base_models(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model_pool: list[ModelConfig],
    folds: list[tuple[np.ndarray, np.ndarray]],
    task_type: str,
    performance_metric: str | None = None,
    return_probabilities: bool = False,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, np.ndarray | None]:
    """Train every model of the pool with k-fold CV and collect out-of-fold predictions.

    For each model in the pool and each CV fold (see :func:`_fit_model_cv`):
      1. Select features based on model's feature_indices
      2. Fit preprocessing pipeline on training data ONLY
      3. Transform training and validation data
      4. Train model on preprocessed training data
      5. Predict on preprocessed validation data

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Feature matrix (n_samples, n_features).
    y : pd.Series or np.ndarray
        Target values (n_samples,).
    model_pool : list[ModelConfig]
        List of model configurations to train.
    folds : list[tuple[np.ndarray, np.ndarray]]
        CV fold splits as (train_indices, val_indices).
    task_type : str
        'classification' or 'regression'.
    performance_metric : str or None
        Metric reported per fold; ``None``/"auto" selects macro-F1 / RMSE.
    return_probabilities : bool
        Also return the out-of-fold class probabilities (classification only;
        ``None`` for regression). Used as level-0 inputs of stacking.
    n_jobs : int
        Number of models trained in parallel (joblib; ``-1`` = all cores). The results
        are identical for every value because each model has a fixed random_state.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        predictions_df : (n_samples, n_models) out-of-fold predictions; columns are
            the model names (``ModelConfig.name``).
        details_df : per-model, per-fold metric values and timings.
        probabilities : (n_samples, n_models, n_classes) out-of-fold class
            probabilities, only when ``return_probabilities`` is True.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n_samples = X_arr.shape[0]
    n_models = len(model_pool)

    # Resolve the metric name once so it appears consistently in outputs
    performance_metric = resolve_metric(performance_metric, task_type)

    # Initialize predictions array with NaN
    predictions = np.full((n_samples, n_models), np.nan)
    collect_proba = return_probabilities and task_type == "classification"
    probabilities: np.ndarray | None = None
    n_classes = 0
    if collect_proba:
        n_classes = int(np.max(y_arr)) + 1  # labels are encoded 0..K-1
        probabilities = np.full((n_samples, n_models, n_classes), np.nan)

    # Collect detailed per-fold metrics
    details_records: list[dict[str, Any]] = []

    task_args = (X_arr, y_arr, folds, task_type, performance_metric,
                 n_classes if collect_proba else None)
    if n_jobs == 1:
        results = (_fit_model_cv(model_cfg, *task_args) for model_cfg in model_pool)
    else:
        # Every model has a fixed random_state, so the output does not depend on n_jobs.
        # joblib (loky) limits the BLAS/OpenMP threads of each worker to avoid oversubscription.
        results = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_fit_model_cv)(model_cfg, *task_args) for model_cfg in model_pool
        )

    if tqdm is not None:
        results = tqdm(results, total=n_models, desc="Training base models", unit="model")

    for model_idx, (model_pred, model_proba, records) in enumerate(results):
        predictions[:, model_idx] = model_pred
        if collect_proba:
            probabilities[:, model_idx, :] = model_proba
        details_records.extend(records)

    predictions_df = pd.DataFrame(predictions, columns=[mc.name for mc in model_pool])

    # Build details DataFrame
    details_df = pd.DataFrame(details_records)

    # Validate no NaN on the rows covered by a test fold (rows never tested stay NaN
    # by design, e.g. the training rows of a single refit "fold").
    covered = np.unique(np.concatenate([test_idx for _, test_idx in folds]))
    nan_count = int(predictions_df.iloc[covered].isna().sum().sum())
    if nan_count > 0:
        logger.warning(
            "Found %d NaN values in predictions — possible CV coverage issue.",
            nan_count,
        )
    else:
        logger.info(
            "All out-of-fold predictions filled. Shape: %s", predictions_df.shape,
        )

    if return_probabilities:
        return predictions_df, details_df, probabilities
    return predictions_df, details_df

