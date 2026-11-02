"""Dataset loading and cross-validation split generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from prunestack.data.task_detection import detect_task_type

logger = logging.getLogger(__name__)


@dataclass
class DatasetBundle:
    """Container for a loaded dataset with CV split information.

    Attributes
    ----------
    X : pd.DataFrame
        Feature matrix (all columns numeric after encoding).
    y : pd.Series
        Target values.
    metrics_df : pd.DataFrame
        Loaded metrics.csv (normalized).
    folds : list[tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) for each fold.
    task_type : str
        Detected or overridden task type.
    dataset_path : Path
        Source directory path.
    label_encoder : LabelEncoder | None
        Label encoder used for classification (None for regression).
    feature_encoders : dict[str, LabelEncoder]
        Mapping of column name -> LabelEncoder for categorical features.
        Empty dict if no categorical features were encoded.
    feature_names : list[str]
        List of feature column names.
    dropped_columns : dict[str, str]
        Removed feature columns and the reason (``constant``, ``identifier``, ``user``).
    n_features : int
        Number of features.
    n_samples : int
        Number of samples.
    """

    X: pd.DataFrame
    y: pd.Series
    metrics_df: pd.DataFrame
    folds: list[tuple[np.ndarray, np.ndarray]]
    task_type: str
    dataset_path: Path
    label_encoder: LabelEncoder | None
    feature_encoders: dict[str, LabelEncoder]
    feature_names: list[str]
    n_features: int
    n_samples: int
    dropped_columns: dict[str, str] = field(default_factory=dict)


# A non-numeric column with at least this share of distinct values is treated as a
# row identifier (names, e-mails, free text) and removed: label-encoded it is an
# arbitrary integer per row, which carries no generalisable information.
IDENTIFIER_UNIQUE_RATIO = 0.9


def find_uninformative_columns(X: pd.DataFrame) -> dict[str, str]:
    """Constant columns and non-numeric identifier columns, with the reason."""
    numeric = set(X.select_dtypes(include=[np.number, "bool"]).columns)
    found: dict[str, str] = {}
    for col in X.columns:
        n_unique = X[col].nunique(dropna=False)
        if n_unique <= 1:
            found[col] = "constant"
        elif col not in numeric and n_unique >= IDENTIFIER_UNIQUE_RATIO * len(X):
            found[col] = "identifier"
    return found


def _validate_dataset_files(dataset_dir: Path) -> tuple[Path, Path, Path]:
    """Validate that required files exist and return their paths."""
    x_path = dataset_dir / "X_test.csv"
    y_path = dataset_dir / "y_test.csv"
    metrics_path = dataset_dir / "metrics.csv"

    missing = []
    for p, name in [(x_path, "X_test.csv"), (y_path, "y_test.csv"), (metrics_path, "metrics.csv")]:
        if not p.exists():
            missing.append(name)

    if missing:
        raise FileNotFoundError(
            f"Missing required files in {dataset_dir}: {', '.join(missing)}"
        )

    return x_path, y_path, metrics_path


def _read_metrics_csv(metrics_path: Path) -> pd.DataFrame:
    """Read ``metrics.csv`` with lower-cased column names and stripped row labels.

    The file is only used to infer the task type from its metric column names.
    """
    df = pd.read_csv(metrics_path, index_col=0)
    df.columns = [str(col).strip().lower() for col in df.columns]
    df.index = [str(idx).strip() for idx in df.index]
    df.index.name = "algorithm"
    return df


def load_dataset(
    dataset_path: str | Path,
    task_type: str = "auto",
    n_folds: int = 5,
    seed: int = 42,
    drop_columns: list[str] | None = None,
) -> DatasetBundle:
    """Load a dataset and prepare cross-validation splits.

    Parameters
    ----------
    dataset_path : str or Path
        Path to dataset directory containing X_test.csv, y_test.csv, metrics.csv.
    task_type : str
        'auto', 'classification', or 'regression'. If 'auto', detected from metrics.
    n_folds : int
        Number of CV folds.
    seed : int
        Random seed for reproducibility.
    drop_columns : list[str] or None
        Feature columns removed on request (e.g. contact strings that are not caught
        by the automatic rules). Constant and identifier columns
        (:func:`find_uninformative_columns`) are always removed.

    Returns
    -------
    DatasetBundle
        Loaded dataset with CV splits and metadata.
    """
    dataset_dir = Path(dataset_path)
    x_path, y_path, metrics_path = _validate_dataset_files(dataset_dir)

    # Load data
    X = pd.read_csv(x_path, index_col=0)
    y_df = pd.read_csv(y_path, index_col=0)
    y = y_df.iloc[:, 0]  # Take the first (and presumably only) column

    logger.info("Loaded X: %s, y: %s from %s", X.shape, y.shape, dataset_dir)

    unknown = sorted(set(drop_columns or []) - set(X.columns))
    if unknown:
        raise ValueError(f"drop_columns not found in {x_path}: {unknown}")
    dropped = {col: "user" for col in drop_columns or []}
    dropped.update({c: r for c, r in find_uninformative_columns(X).items() if c not in dropped})
    if dropped:
        X = X.drop(columns=list(dropped))
        logger.info("Dropped %d feature column(s): %s", len(dropped), dropped)
    if X.shape[1] == 0:
        raise ValueError(f"No informative feature columns left in {x_path} (dropped: {dropped})")

    # Encode non-numeric feature columns (pandas 3 'str' dtype or 'object') as integers.
    feature_encoders: dict[str, LabelEncoder] = {}
    numeric_cols = set(X.select_dtypes(include=[np.number, "bool"]).columns)
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    if cat_cols:
        logger.info(
            "Detected %d categorical feature column(s): %s",
            len(cat_cols),
            cat_cols,
        )
        for col in cat_cols:
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col].astype(str))
            feature_encoders[col] = enc
            logger.debug(
                "Encoded feature '%s': %s -> %s",
                col,
                list(enc.classes_),
                list(range(len(enc.classes_))),
            )
        logger.info("Encoded %d categorical feature(s) to integers.", len(cat_cols))

    # Validate alignment
    if len(X) != len(y):
        raise ValueError(
            f"X and y have different lengths: X={len(X)}, y={len(y)}"
        )

    metrics_df = _read_metrics_csv(metrics_path)
    resolved_task_type = detect_task_type(metrics_df, override=task_type, y=y)

    # Encode labels for classification if needed
    label_encoder = None
    if resolved_task_type == "classification":
        # Always apply LabelEncoder for classification to handle:
        # - string labels (e.g., "cat", "dog")
        # - non-sequential integer labels (e.g., 1, 3, 5)
        # - mixed-type labels
        unique_labels = y.unique()
        try:
            sorted_unique = sorted(unique_labels)
        except TypeError:
            # Mixed types that can't be sorted - force string conversion
            y = y.astype(str)
            sorted_unique = sorted(y.unique())

        # Check if labels are already 0-indexed contiguous integers
        is_zero_indexed = False
        try:
            as_ints = [int(v) for v in sorted_unique]
            is_zero_indexed = as_ints == list(range(len(as_ints)))
        except (ValueError, TypeError):
            pass

        if not is_zero_indexed:
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index, name=y.name)
            logger.info(
                "Applied LabelEncoder: %s -> %s",
                label_encoder.classes_,
                list(range(len(label_encoder.classes_))),
            )
        else:
            # Ensure integer dtype
            y = y.astype(int)
    else:
        # Regression: ensure numeric target
        y = pd.to_numeric(y, errors="coerce")
        if y.isna().any():
            n_nan = int(y.isna().sum())
            logger.warning(
                "Regression target contains %d non-numeric values (converted to NaN). "
                "Dropping those rows.", n_nan,
            )
            valid_mask = ~y.isna()
            y = y[valid_mask]
            X = X.loc[valid_mask]

    # Edge case: auto-adjust n_folds for small datasets
    n_samples = len(X)
    effective_folds = n_folds
    if resolved_task_type == "classification":
        min_class_count = int(pd.Series(y).value_counts().min())
        if min_class_count < n_folds:
            effective_folds = max(2, min_class_count)
            logger.warning(
                "Smallest class has only %d samples — reducing n_folds from %d to %d.",
                min_class_count, n_folds, effective_folds,
            )
    if n_samples < effective_folds * 2:
        effective_folds = max(2, n_samples // 2)
        logger.warning(
            "Dataset too small (%d samples) for %d folds — reduced to %d folds.",
            n_samples, n_folds, effective_folds,
        )
    if n_samples < 50:
        logger.warning(
            "Very small dataset (%d samples). Results may not be reliable.",
            n_samples,
        )

    # Create CV splits
    if resolved_task_type == "classification":
        cv = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        folds = list(cv.split(X, y))
    else:
        cv = KFold(n_splits=effective_folds, shuffle=True, random_state=seed)
        folds = list(cv.split(X))

    logger.info(
        "Created %d-fold CV splits (%s) for %s task",
        effective_folds,
        "stratified" if resolved_task_type == "classification" else "standard",
        resolved_task_type,
    )

    feature_names = list(X.columns)

    return DatasetBundle(
        X=X,
        y=y,
        metrics_df=metrics_df,
        folds=folds,
        task_type=resolved_task_type,
        dataset_path=dataset_dir,
        label_encoder=label_encoder,
        feature_encoders=feature_encoders,
        feature_names=feature_names,
        n_features=X.shape[1],
        n_samples=X.shape[0],
        dropped_columns=dropped,
    )
