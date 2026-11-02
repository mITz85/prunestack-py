"""Task type detection from metrics.csv column names and label analysis."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Metric column sets that identify each task type
CLASSIFICATION_INDICATORS = {"accuracy", "f1", "precision", "recall", "kappa"}
REGRESSION_INDICATORS = {"mae", "mse", "rmse", "r2"}

TaskType = Literal["classification", "regression"]


def detect_task_from_labels(y: pd.Series | np.ndarray) -> TaskType | None:
    """Infer task type from the target labels as a fallback heuristic.

    Parameters
    ----------
    y : array-like
        Target values.

    Returns
    -------
    TaskType or None
        'classification' or 'regression' if confidently detected, else None.
    """
    y = np.asarray(y)
    n_unique = len(np.unique(y))

    # String labels → classification
    if y.dtype.kind in ("U", "S", "O"):
        # Object or string dtype
        try:
            # Check if all can convert to float
            y_float = y.astype(float)
        except (ValueError, TypeError):
            logger.info("Label-based detection: string labels → classification")
            return "classification"

    # Very few unique values relative to sample count → classification
    if n_unique <= 20 and n_unique < len(y) * 0.05:
        # Check if integer-like
        try:
            y_float = y.astype(float)
            if np.all(y_float == np.floor(y_float)):
                logger.info(
                    "Label-based detection: %d unique integer-like values → classification",
                    n_unique,
                )
                return "classification"
        except (ValueError, TypeError):
            pass

    # Many unique float values → regression
    try:
        y_float = y.astype(float)
        if n_unique > 20 or n_unique > len(y) * 0.5:
            logger.info(
                "Label-based detection: %d unique continuous values → regression",
                n_unique,
            )
            return "regression"
    except (ValueError, TypeError):
        pass

    return None


def detect_task_type(
    metrics_df: pd.DataFrame,
    override: str = "auto",
    y: pd.Series | np.ndarray | None = None,
) -> TaskType:
    """Detect whether a dataset is classification or regression.

    Detection priority:
    1. Explicit override ('classification' or 'regression')
    2. Metrics column names in metrics_df
    3. Label-based heuristic (if y is provided)

    Parameters
    ----------
    metrics_df : pd.DataFrame
        The loaded metrics.csv DataFrame with metric columns.
    override : str
        If 'classification' or 'regression', skip auto-detection.
        If 'auto' (default), detect from column names then labels.
    y : Series or ndarray, optional
        Target values for label-based fallback detection.

    Returns
    -------
    TaskType
        Either 'classification' or 'regression'.

    Raises
    ------
    ValueError
        If task type cannot be determined (ambiguous or missing indicators).
    """
    if override in ("classification", "regression"):
        logger.info("Task type override: %s", override)
        return override  # type: ignore[return-value]

    if override != "auto":
        raise ValueError(
            f"Invalid task_type override: '{override}'. "
            "Must be 'auto', 'classification', or 'regression'."
        )

    # Normalize column names to lowercase, stripped
    columns = {col.strip().lower() for col in metrics_df.columns}

    has_classification = bool(columns & CLASSIFICATION_INDICATORS)
    has_regression = bool(columns & REGRESSION_INDICATORS)

    if has_classification and has_regression:
        raise ValueError(
            "Ambiguous task type: metrics.csv contains both classification "
            f"indicators {columns & CLASSIFICATION_INDICATORS} and regression "
            f"indicators {columns & REGRESSION_INDICATORS}. "
            "Use --task-type to override."
        )

    if has_classification:
        detected = "classification"
    elif has_regression:
        detected = "regression"
    else:
        # Fallback: try label-based detection
        if y is not None:
            label_detected = detect_task_from_labels(y)
            if label_detected is not None:
                logger.info(
                    "Task type detected from labels (metrics columns inconclusive): %s",
                    label_detected,
                )
                return label_detected  # type: ignore[return-value]

        raise ValueError(
            "Cannot detect task type: metrics.csv columns "
            f"{columns} contain neither classification indicators "
            f"{CLASSIFICATION_INDICATORS} nor regression indicators "
            f"{REGRESSION_INDICATORS}. Use --task-type to override."
        )

    logger.info(
        "Auto-detected task type: %s (from columns: %s)",
        detected,
        columns & (CLASSIFICATION_INDICATORS | REGRESSION_INDICATORS),
    )
    return detected  # type: ignore[return-value]
