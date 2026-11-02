"""Preprocessing pipeline for base model training (no-leakage design)."""

from __future__ import annotations

import logging

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def create_preprocessing_pipeline() -> Pipeline:
    """Create a preprocessing pipeline: impute missing values then scale.

    The pipeline consists of:
      1. SimpleImputer(strategy='median') — fill NaN with median
      2. StandardScaler() — zero mean, unit variance

    IMPORTANT: This pipeline must be fitted ONLY on training data within
    each CV fold to prevent data leakage.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted preprocessing pipeline.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    logger.debug("Created preprocessing pipeline: imputer(median) -> scaler")
    return pipeline
