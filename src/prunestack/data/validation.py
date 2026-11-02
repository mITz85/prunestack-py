"""Input validation and data quality checks for prunestack.

Provides functions to validate datasets, base-learner hyperparameters, and
experiment configurations before the pipeline runs, giving users
clear feedback on problems and suggestions for fixes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Container for validation outcomes.

    Attributes
    ----------
    is_valid : bool
        True if no errors were found (warnings are OK).
    errors : list[str]
        Fatal issues that prevent the pipeline from running.
    warnings : list[str]
        Non-fatal issues the user should be aware of.
    info : list[str]
        Informational messages about data characteristics.
    """

    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        self.info.append(msg)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another ValidationResult into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.is_valid:
            self.is_valid = False

    def log_all(self) -> None:
        """Emit all messages through the logging system."""
        for msg in self.info:
            logger.info("  ℹ %s", msg)
        for msg in self.warnings:
            logger.warning("  ⚠ %s", msg)
        for msg in self.errors:
            logger.error("  ✗ %s", msg)

    def summary(self) -> str:
        """Return a one-line summary."""
        parts = []
        if self.errors:
            parts.append(f"{len(self.errors)} error(s)")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warning(s)")
        if not parts:
            return "Validation passed ✓"
        return "Validation: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_dataset(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    dataset_path: str | Path | None = None,
) -> ValidationResult:
    """Validate dataset quality and characteristics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    dataset_path : str or Path, optional
        Dataset directory (for informational messages).

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    # Convert X to a numeric array, encoding any remaining string columns first.
    # We normalise everything to a DataFrame so the same encoding path handles
    # both pandas DataFrames and numpy arrays (including object/str dtype arrays).
    if isinstance(X, pd.DataFrame):
        X_df = X
    elif isinstance(X, np.ndarray):
        # numpy arrays with non-numeric dtype (object, str/unicode '<U..', bytes 'S..')
        # cannot be cast straight to float, so wrap in a DataFrame for encoding.
        if X.dtype.kind in ("O", "U", "S"):
            X_df = pd.DataFrame(X)
        else:
            # Already numeric — fast path, no DataFrame round-trip needed.
            X_arr = X.astype(float)
            X_df = None
    else:
        X_df = pd.DataFrame(np.asarray(X))

    if X_df is not None:
        # Detect non-numeric (categorical/string) columns as the complement of the
        # numeric columns. This is forward compatible with pandas 3/4 where string
        # columns may use the dedicated 'str' dtype instead of 'object' — selecting
        # 'str' directly raises in pandas 2.x, so we invert the numeric selection.
        numeric_cols = set(X_df.select_dtypes(include=[np.number, "bool"]).columns)
        cat_cols = [c for c in X_df.columns if c not in numeric_cols]
        if cat_cols:
            result.add_warning(
                f"{len(cat_cols)} categorical feature column(s) detected in validation: "
                f"{cat_cols[:5]}{'...' if len(cat_cols) > 5 else ''}. "
                "These should be encoded before reaching validation. "
                "Converting to numeric codes for validation checks."
            )
            X_numeric = X_df.copy()
            for col in cat_cols:
                X_numeric[col] = pd.Categorical(X_numeric[col]).codes
            X_arr = np.asarray(X_numeric, dtype=float)
        else:
            X_arr = np.asarray(X_df, dtype=float)

    y_arr = np.asarray(y)

    n_samples, n_features = X_arr.shape

    # --- shape checks ---
    if n_samples == 0:
        result.add_error("Dataset is empty (0 samples).")
        return result

    if n_features == 0:
        result.add_error("Dataset has no features (0 columns).")
        return result

    if len(y_arr) != n_samples:
        result.add_error(
            f"Shape mismatch: X has {n_samples} samples but y has {len(y_arr)}."
        )
        return result

    result.add_info(f"Dataset: {n_samples} samples × {n_features} features")

    # --- missing values ---
    n_missing = int(np.isnan(X_arr).sum())
    if n_missing > 0:
        pct = 100 * n_missing / X_arr.size
        result.add_warning(
            f"Missing values detected: {n_missing} ({pct:.1f}% of cells). "
            "The preprocessing pipeline will impute with median."
        )
    else:
        result.add_info("No missing values detected.")

    # --- constant features (zero variance) ---
    variances = np.nanvar(X_arr, axis=0)
    constant_mask = variances == 0
    n_constant = int(constant_mask.sum())
    if n_constant > 0:
        result.add_warning(
            f"{n_constant} constant feature(s) detected (zero variance). "
            "These add no predictive power and may slow training."
        )

    # --- duplicate samples ---
    try:
        if isinstance(X, pd.DataFrame):
            n_dup = int(X.duplicated().sum())
        else:
            _, counts = np.unique(X_arr, axis=0, return_counts=True)
            n_dup = int((counts > 1).sum())
        if n_dup > 0:
            result.add_warning(f"{n_dup} duplicate sample(s) detected.")
    except (TypeError, ValueError):
        pass  # can't check duplicates for unusual dtypes

    # --- high dimensionality ---
    if n_features > n_samples:
        result.add_warning(
            f"High-dimensional data: {n_features} features > {n_samples} samples. "
            "Consider feature selection or dimensionality reduction."
        )

    # --- target analysis ---
    unique_y = np.unique(y_arr)
    n_classes = len(unique_y)

    if n_classes <= 20 and np.issubdtype(y_arr.dtype, np.integer):
        # Classification target
        result.add_info(f"Target appears to be classification with {n_classes} classes.")
        class_counts = pd.Series(y_arr).value_counts()
        min_count = int(class_counts.min())
        max_count = int(class_counts.max())
        imbalance_ratio = max_count / max(min_count, 1)
        if imbalance_ratio > 10:
            result.add_warning(
                f"Severe class imbalance detected (ratio {imbalance_ratio:.1f}:1). "
                "Consider using stratified CV (default) and balanced metrics."
            )
        elif imbalance_ratio > 3:
            result.add_warning(
                f"Moderate class imbalance detected (ratio {imbalance_ratio:.1f}:1)."
            )
        else:
            result.add_info(f"Class balance: ratio {imbalance_ratio:.1f}:1 (acceptable).")
    else:
        # Regression target
        result.add_info(
            f"Target appears continuous (regression): "
            f"range [{y_arr.min():.4f}, {y_arr.max():.4f}], "
            f"mean={y_arr.mean():.4f}, std={y_arr.std():.4f}"
        )

    # --- small dataset warning ---
    if n_samples < 50:
        result.add_warning(
            f"Very small dataset ({n_samples} samples). "
            "Results may not be reliable. Consider reducing n_folds."
        )

    if dataset_path:
        result.add_info(f"Dataset path: {dataset_path}")

    return result


# ---------------------------------------------------------------------------
# Base-learner hyperparameter validation
# ---------------------------------------------------------------------------

def validate_hyperparameters(
    hyperparameters: dict[str, dict[str, Any]],
    task_type: str,
) -> ValidationResult:
    """Validate the fixed base-model hyperparameters loaded from the JSON configuration.

    Parameters
    ----------
    hyperparameters : dict
        Mapping of algorithm name -> keyword arguments of the estimator.
    task_type : str
        'classification' or 'regression'.
    """
    from prunestack.models.registry import list_algorithms

    known_algorithms = set(list_algorithms(task_type if task_type != "auto" else None))
    result = ValidationResult()

    if not hyperparameters:
        result.add_error(
            "No base-learner hyperparameters found. Ensure the hyperparameter "
            "configuration file (configs/hyperparameters.json or "
            "--hyperparameters PATH) contains a section for this task type."
        )
        return result

    n_algos = len(hyperparameters)
    result.add_info(f"Found fixed hyperparameters for {n_algos} algorithm(s): {sorted(hyperparameters)}")

    if n_algos < 2:
        result.add_warning(
            f"Only {n_algos} algorithm(s) found. Diversity metrics require "
            "at least 2 algorithms for meaningful results."
        )

    if n_algos < 6:
        result.add_warning(
            f"Fewer than 6 algorithms ({n_algos}). "
            "The experiment pool expects 6 algorithms × 20 variants = 120 models."
        )

    for algo, params in hyperparameters.items():
        if algo not in known_algorithms:
            result.add_error(f"Algorithm '{algo}' is not registered for {task_type}.")

        if not isinstance(params, dict):
            result.add_error(f"Hyperparameters for '{algo}' must be a dict, got {type(params).__name__}.")
            continue

        # Check for problematic parameter values
        for k, v in params.items():
            if v is None:
                result.add_warning(f"{algo}.{k} is None — will use sklearn default.")

    return result


# ---------------------------------------------------------------------------
# Config validation (extended)
# ---------------------------------------------------------------------------

def validate_config(config: Any) -> ValidationResult:
    """Validate an ExperimentConfig for common issues beyond basic type checks.

    The dataclass __post_init__ already does range validation; this function
    performs deeper semantic checks and produces user-friendly messages.

    Parameters
    ----------
    config : ExperimentConfig

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    # α grid values
    if config.alpha_grid is not None:
        for v in config.alpha_grid:
            if not (0.0 <= v <= 1.0):
                result.add_error(
                    f"alpha_grid value {v} out of range [0, 1]. "
                    "Alpha controls the performance-diversity trade-off weight."
                )

    # Seed
    if config.seed < 0:
        result.add_error(f"Seed must be non-negative, got {config.seed}.")

    # Path existence
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        result.add_error(f"Dataset path does not exist: {dataset_path}")
    else:
        # Check required files
        for fname in ["X_test.csv", "y_test.csv", "metrics.csv"]:
            if not (dataset_path / fname).exists():
                result.add_error(f"Required file missing: {dataset_path / fname}")

    # EHO optimizer params
    if config.optimizer == "eho":
        pop_size = config.eho_n_clans * config.eho_clan_size
        result.add_info(
            f"EHO population size: {pop_size} "
            f"({config.eho_n_clans} clans × {config.eho_clan_size} elephants)"
        )
        if pop_size > 500:
            result.add_warning(
                f"Large EHO population ({pop_size}). This may be slow."
            )

    # n_folds vs data (can only warn generically here)
    if config.n_folds > 10:
        result.add_warning(
            f"n_folds={config.n_folds} is unusually high. "
            "Consider 5 or 10 folds for most datasets."
        )

    return result


# ---------------------------------------------------------------------------
# Convenience: run all validations at pipeline start
# ---------------------------------------------------------------------------

def run_all_validations(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    hyperparameters: dict[str, dict[str, Any]],
    config: Any,
    task_type: str | None = None,
    strict: bool = False,
) -> ValidationResult:
    """Run dataset, hyperparameter, and config validations.

    Parameters
    ----------
    task_type : str or None
        Resolved task type; ``None`` -> ``config.task_type``.
    strict : bool
        If True, treat warnings as errors.

    Returns
    -------
    ValidationResult
        Merged result from all checks.
    """
    from prunestack.utils.exceptions import DataValidationError

    combined = ValidationResult()

    combined.merge(validate_dataset(X, y, config.dataset_path))
    combined.merge(validate_hyperparameters(hyperparameters, task_type or config.task_type))
    combined.merge(validate_config(config))

    # In strict mode, promote warnings to errors
    if strict and combined.warnings:
        for w in combined.warnings:
            combined.add_error(f"[strict] {w}")

    combined.log_all()
    logger.info(combined.summary())

    if not combined.is_valid:
        raise DataValidationError(
            f"Validation failed with {len(combined.errors)} error(s). "
            "See log messages above for details.",
            suggestion="Fix the reported errors and re-run.",
        )

    return combined
