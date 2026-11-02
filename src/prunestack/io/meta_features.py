"""Meta-feature extraction for the meta-learning layer.

Meta-features (inputs of the meta-learner) form three groups, all computable
*before* ensemble pruning is run:

1. dataset characteristics (simple, statistical, information-theoretic,
   complexity and landmarking measures),
2. base-model characteristics of the out-of-fold model pool,
3. ensemble characteristics (diversity, oracle, top-/bottom-k ensembles).

A fourth block, the *optimization summary* (:func:`extract_optimization_summary`),
describes the EHO results over the alpha grid. It is derived from the outcome the
meta-learner is meant to predict (e.g. ``opt_best_alpha`` is the best alpha itself),
so it is target leakage and is never part of the meta-features; it is written to a
separate file for analysis only.

Performance-based meta-features use the experiment's primary metric on the
[0, 1] fitness scale of :func:`prunestack.metrics.performance.normalize_performance_score`:
macro-F1 for classification and the RMSE skill score ``1 - RMSE / sd(y)`` for
regression (scale-free, hence comparable across datasets).

Features that were exact copies or affine functions of a single other feature
were removed (see ``docs/meta_features.md``). The set of feature names is fixed and
identical for both tasks; every value is a float. Values that cannot be computed
(e.g. degenerate inputs) are reported as 0.0 and logged.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold

from prunestack.metrics.performance import (
    calculate_metric,
    evaluate_ensemble_score,
    normalize_performance_score,
    resolve_metric,
)

logger = logging.getLogger(__name__)


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 for NaN/inf."""
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


Folds = list[tuple[np.ndarray, np.ndarray]]


def _score(y_true: np.ndarray, y_pred: np.ndarray, task_type: str, metric: str) -> float:
    """Primary metric of the experiment on the [0, 1] fitness scale (higher is better)."""
    if task_type == "classification":
        y_pred = np.asarray(y_pred).astype(int)
    return normalize_performance_score(calculate_metric(y_true, y_pred, metric), metric, y_true)


def _default_folds(y: np.ndarray, task_type: str, n_folds: int, seed: int) -> Folds:
    """Stratified (classification) or plain shuffled k-fold splits."""
    n_samples = len(y)
    if task_type == "classification":
        min_class = int(np.unique(y, return_counts=True)[1].min())
        k = max(2, min(n_folds, min_class))
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(n_samples), y))
    splitter = KFold(n_splits=max(2, min(n_folds, n_samples // 2)), shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n_samples)))


def _cv_score(estimator: Any, X: np.ndarray, y: np.ndarray, folds: Folds,
              task_type: str, metric: str) -> float:
    """Mean per-fold score (see :func:`_score`) of ``estimator`` over ``folds``."""
    scores = []
    for train_idx, val_idx in folds:
        model = clone(estimator).fit(X[train_idx], y[train_idx])
        scores.append(_score(y[val_idx], model.predict(X[val_idx]), task_type, metric))
    return float(np.mean(scores))


def _extract_dataset_characteristics(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    performance_metric: str | None = None,
    folds: Folds | None = None,
    seed: int = 1985,
) -> dict:
    """Extract 48 dataset-level meta-features.

    Landmarkers and the nearest-neighbour error are cross-validated on ``folds``
    (the experiment's folds; default: 5-fold, stratified for classification) and
    scored with the experiment's primary metric (see :func:`_score`).

    Categories:
      - Simple / statistical
      - Information-theoretic
      - Complexity / geometry
      - Landmarking (CV score of simple learners)
      - Structural
    """
    n_samples, n_features = X.shape
    features: dict[str, float] = {}
    metric = resolve_metric(performance_metric, task_type)
    if folds is None:
        folds = _default_folds(y, task_type, 5, seed)

    # Simple / statistical
    features["n_samples"] = float(n_samples)
    features["n_features"] = float(n_features)

    if task_type == "classification":
        classes, counts = np.unique(y, return_counts=True)
        n_classes = len(classes)
        features["n_classes"] = float(n_classes)
        minority = counts.min()
        majority = counts.max()
        features["class_imbalance_ratio"] = float(minority / majority) if majority > 0 else 0.0
    else:
        features["n_classes"] = 0.0  # regression
        features["class_imbalance_ratio"] = 0.0

    features["feature_to_sample_ratio"] = n_features / n_samples if n_samples > 0 else 0.0

    # Feature correlations
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if n_features > 1:
            corr_matrix = np.corrcoef(X, rowvar=False)
            # Upper triangle (exclude diagonal)
            triu_idx = np.triu_indices(n_features, k=1)
            corr_vals = np.abs(corr_matrix[triu_idx])
            corr_vals = corr_vals[~np.isnan(corr_vals)]
            features["mean_feature_correlation"] = _safe_float(np.mean(corr_vals)) if len(corr_vals) > 0 else 0.0
            features["max_feature_correlation"] = _safe_float(np.max(corr_vals)) if len(corr_vals) > 0 else 0.0
        else:
            features["mean_feature_correlation"] = 0.0
            features["max_feature_correlation"] = 0.0

    features["mean_feature_std"] = _safe_float(np.nanmean(np.nanstd(X, axis=0)))
    features["mean_feature_skewness"] = _safe_float(np.nanmean(sp_stats.skew(X, axis=0, nan_policy="omit")))
    features["mean_feature_kurtosis"] = _safe_float(np.nanmean(sp_stats.kurtosis(X, axis=0, nan_policy="omit")))
    features["missing_value_ratio"] = float(np.sum(np.isnan(X))) / (n_samples * n_features) if (n_samples * n_features) > 0 else 0.0

    # Categorical vs numerical (heuristic: < 10 unique values = categorical)
    n_cat = sum(1 for j in range(n_features) if len(np.unique(X[:, j][~np.isnan(X[:, j])])) < 10)
    features["categorical_feature_ratio"] = n_cat / n_features if n_features > 0 else 0.0

    # Feature entropy
    def _feature_entropy(col: np.ndarray) -> float:
        col_clean = col[~np.isnan(col)]
        if len(col_clean) == 0:
            return 0.0
        # Bin continuous features
        _, counts = np.unique(np.digitize(col_clean, bins=np.histogram_bin_edges(col_clean, bins="auto")), return_counts=True)
        probs = counts / counts.sum()
        return float(sp_stats.entropy(probs))

    feat_entropies = [_feature_entropy(X[:, j]) for j in range(n_features)]
    features["feature_entropy_mean"] = _safe_float(np.mean(feat_entropies))

    if task_type == "classification":
        _, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        features["label_entropy"] = float(sp_stats.entropy(probs))
    else:
        # For regression, use binned entropy
        _, counts = np.unique(np.digitize(y, bins=np.histogram_bin_edges(y, bins="auto")), return_counts=True)
        probs = counts / counts.sum()
        features["label_entropy"] = float(sp_stats.entropy(probs))

    # Information-theoretic
    try:
        X_clean = np.nan_to_num(X, nan=0.0)
        if task_type == "classification":
            mi_values = mutual_info_classif(X_clean, y, random_state=seed, n_neighbors=3)
        else:
            mi_values = mutual_info_regression(X_clean, y, random_state=seed, n_neighbors=3)
        features["mutual_information_mean"] = _safe_float(np.mean(mi_values))
        features["mutual_information_max"] = _safe_float(np.max(mi_values))
        features["mutual_information_std"] = _safe_float(np.std(mi_values))
    except Exception:
        features["mutual_information_mean"] = 0.0
        features["mutual_information_max"] = 0.0
        features["mutual_information_std"] = 0.0

    # Feature-feature MI (sample pairs for efficiency)
    n_pairs = min(n_features * (n_features - 1) // 2, 50)
    if n_features > 1 and n_pairs > 0:
        rng = np.random.RandomState(seed)
        pair_mis = []
        for _ in range(n_pairs):
            i, j = rng.choice(n_features, 2, replace=False)
            try:
                mi_val = mutual_info_regression(X_clean[:, [i]], X_clean[:, j], random_state=seed, n_neighbors=3)
                pair_mis.append(float(mi_val[0]))
            except Exception:
                pair_mis.append(0.0)
        features["feature_mutual_information_mean"] = _safe_float(np.mean(pair_mis))
    else:
        features["feature_mutual_information_mean"] = 0.0

    # Joint entropy, normalized MI, info gain ratio (approximations)
    features["joint_entropy_mean"] = features["feature_entropy_mean"] + features["label_entropy"]
    nmi = features["mutual_information_mean"] / features["label_entropy"] if features["label_entropy"] > 0 else 0.0
    features["normalized_mutual_information_mean"] = _safe_float(nmi)
    igr = features["mutual_information_mean"] / features["feature_entropy_mean"] if features["feature_entropy_mean"] > 0 else 0.0
    features["information_gain_ratio_mean"] = _safe_float(igr)

    # Complexity / geometry
    X_scaled = StandardScaler().fit_transform(np.nan_to_num(X, nan=0.0))

    if task_type == "classification" and len(np.unique(y)) >= 2:
        classes = np.unique(y)
        # Fisher discriminant ratio
        try:
            between_var = np.var([X_scaled[y == c].mean(axis=0) for c in classes], axis=0)
            within_var = np.mean([X_scaled[y == c].var(axis=0) for c in classes], axis=0)
            fdr = np.mean(between_var / (within_var + 1e-10))
            features["fisher_discriminant_ratio"] = _safe_float(fdr)
        except Exception:
            features["fisher_discriminant_ratio"] = 0.0

        # Class separability (centroid distance)
        try:
            centroids = np.array([X_scaled[y == c].mean(axis=0) for c in classes])
            dists = np.linalg.norm(centroids[0] - centroids[1]) if len(classes) == 2 else \
                np.mean([np.linalg.norm(centroids[i] - centroids[j])
                         for i in range(len(classes)) for j in range(i + 1, len(classes))])
            features["class_separability_score"] = _safe_float(dists)
        except Exception:
            features["class_separability_score"] = 0.0
    else:
        features["fisher_discriminant_ratio"] = 0.0
        features["class_separability_score"] = 0.0

    # Volume of overlap (approximation via feature overlap)
    if task_type == "classification" and len(np.unique(y)) >= 2:
        overlaps = []
        classes = np.unique(y)
        for j in range(n_features):
            ranges = [(X_scaled[y == c, j].min(), X_scaled[y == c, j].max()) for c in classes]
            overlap = max(0, min(r[1] for r in ranges) - max(r[0] for r in ranges))
            total = max(r[1] for r in ranges) - min(r[0] for r in ranges)
            overlaps.append(overlap / (total + 1e-10))
        features["volume_of_overlap"] = _safe_float(np.mean(overlaps))
    else:
        features["volume_of_overlap"] = 0.0

    # Feature efficiency
    mi_threshold = 0.01
    try:
        mi_vals = mutual_info_classif(X_clean, y, random_state=seed, n_neighbors=3) if task_type == "classification" \
            else mutual_info_regression(X_clean, y, random_state=seed, n_neighbors=3)
        features["feature_efficiency"] = float(np.mean(mi_vals > mi_threshold))
    except Exception:
        features["feature_efficiency"] = 0.0

    features["dimensionality_curse_indicator"] = n_features / np.log(n_samples + 1)

    # Nearest-neighbour error rate: 1 - cross-validated score of 1-NN
    try:
        nn = KNeighborsClassifier(n_neighbors=1) if task_type == "classification" \
            else KNeighborsRegressor(n_neighbors=1)
        features["nearest_neighbor_error_rate"] = 1.0 - _safe_float(
            _cv_score(nn, X_scaled, y, folds, task_type, metric))
    except Exception:
        features["nearest_neighbor_error_rate"] = 0.5

    # Decision tree depth
    try:
        if task_type == "classification":
            dt = DecisionTreeClassifier(random_state=seed)
        else:
            dt = DecisionTreeRegressor(random_state=seed)
        dt.fit(X_scaled, y)
        features["decision_tree_depth"] = float(dt.get_depth())
    except Exception:
        features["decision_tree_depth"] = 0.0

    # PCA components for 95% variance
    try:
        pca = PCA(random_state=seed)
        pca.fit(X_scaled)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n95 = int(np.searchsorted(cumvar, 0.95) + 1)
        features["pca_explained_variance_95"] = float(min(n95, n_features))
    except Exception:
        features["pca_explained_variance_95"] = float(n_features)

    # Intrinsic dimensionality (MLE estimate)
    try:
        from sklearn.neighbors import NearestNeighbors
        k = min(10, n_samples - 1)
        nn_model = NearestNeighbors(n_neighbors=k + 1)
        nn_model.fit(X_scaled)
        dists, _ = nn_model.kneighbors(X_scaled)
        dists = dists[:, 1:]  # exclude self
        dists = dists[dists[:, -1] > 0]  # remove zero-distance rows
        if len(dists) > 0:
            log_ratios = np.log(dists[:, -1:] / (dists[:, :-1] + 1e-10))
            id_est = float(1.0 / np.mean(log_ratios))
            features["intrinsic_dimensionality_estimate"] = _safe_float(abs(id_est))
        else:
            features["intrinsic_dimensionality_estimate"] = float(n_features)
    except Exception:
        features["intrinsic_dimensionality_estimate"] = float(n_features)

    # Landmarkers: cross-validated score of simple, fast learners on the
    # experiment's folds, scored with the primary metric (macro-F1 / RMSE skill
    # score). They are dataset descriptors only and not part of the model pool.
    landmarkers = {
        # Naive Bayes has no regression counterpart (Ridge would duplicate
        # "linear_model"), so it is reported as 0.0 for regression.
        "naive_bayes": (GaussianNB(), None),
        "decision_stump_depth3": (DecisionTreeClassifier(max_depth=3, random_state=seed),
                                  DecisionTreeRegressor(max_depth=3, random_state=seed)),
        "linear_svm": (LinearSVC(max_iter=1000, random_state=seed), LinearSVR(max_iter=1000, random_state=seed)),
        "knn3": (KNeighborsClassifier(n_neighbors=3), KNeighborsRegressor(n_neighbors=3)),
        "random_forest10": (RandomForestClassifier(n_estimators=10, random_state=seed),
                            RandomForestRegressor(n_estimators=10, random_state=seed)),
        "linear_model": (LogisticRegression(max_iter=1000, random_state=seed), Ridge()),
    }
    for name, (classifier, regressor) in landmarkers.items():
        model = classifier if task_type == "classification" else regressor
        if model is None:  # landmarker not defined for this task
            features[f"landmark_{name}_score"] = 0.0
            continue
        try:
            features[f"landmark_{name}_score"] = _safe_float(
                _cv_score(model, X_scaled, y, folds, task_type, metric))
        except Exception:
            features[f"landmark_{name}_score"] = 0.0

    # Structural
    features["log_n_samples"] = float(np.log(n_samples + 1))
    features["log_n_features"] = float(np.log(n_features + 1))
    features["sample_feature_product"] = float(n_samples * n_features)
    features["sparsity_ratio"] = float(np.sum(X == 0)) / (n_samples * n_features) if (n_samples * n_features) > 0 else 0.0

    # Feature range statistics
    feat_ranges = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
    features["mean_feature_range"] = _safe_float(np.mean(feat_ranges))
    features["std_feature_range"] = _safe_float(np.std(feat_ranges))
    features["max_feature_range"] = _safe_float(np.max(feat_ranges))
    features["min_feature_range"] = _safe_float(np.min(feat_ranges))

    # Outlier ratio (> 3 std from mean)
    z_scores = np.abs(X_scaled)
    features["outlier_ratio"] = float(np.mean(z_scores > 3.0))

    # Target statistics
    features["target_mean"] = _safe_float(np.mean(y.astype(float)))
    features["target_std"] = _safe_float(np.std(y.astype(float)))
    features["target_skewness"] = _safe_float(sp_stats.skew(y.astype(float)))

    return features


def _extract_base_model_characteristics(
    predictions_matrix: np.ndarray,
    y_true: np.ndarray,
    task_type: str,
    performance_metric: str | None = None,
) -> dict:
    """Base-model characteristics (35): individual performance, prediction statistics, agreement.

    Individual performance uses the experiment's primary metric (see :func:`_score`).
    """
    n_samples, n_models = predictions_matrix.shape
    features: dict[str, float] = {}
    metric = resolve_metric(performance_metric, task_type)

    # Individual model performances
    perfs = [_score(y_true, predictions_matrix[:, i], task_type, metric) for i in range(n_models)]

    perfs_arr = np.array(perfs)
    features["model_perf_mean"] = _safe_float(np.mean(perfs_arr))
    features["model_perf_std"] = _safe_float(np.std(perfs_arr))
    features["model_perf_max"] = _safe_float(np.max(perfs_arr))
    features["model_perf_min"] = _safe_float(np.min(perfs_arr))
    features["model_perf_median"] = _safe_float(np.median(perfs_arr))
    features["model_perf_range"] = features["model_perf_max"] - features["model_perf_min"]
    features["model_perf_iqr"] = _safe_float(np.percentile(perfs_arr, 75) - np.percentile(perfs_arr, 25))
    features["model_perf_skew"] = _safe_float(sp_stats.skew(perfs_arr))
    features["model_perf_kurtosis"] = _safe_float(sp_stats.kurtosis(perfs_arr))
    features["n_models_above_mean"] = float(np.sum(perfs_arr > np.mean(perfs_arr)))

    # Prediction statistics
    pred_means = np.mean(predictions_matrix, axis=0)
    pred_stds = np.std(predictions_matrix, axis=0)
    features["pred_mean_of_means"] = _safe_float(np.mean(pred_means))
    features["pred_std_of_means"] = _safe_float(np.std(pred_means))
    features["pred_mean_of_stds"] = _safe_float(np.mean(pred_stds))
    features["pred_std_of_stds"] = _safe_float(np.std(pred_stds))
    features["pred_range_mean"] = _safe_float(np.mean(np.ptp(predictions_matrix, axis=0)))

    # Per-sample prediction variance
    sample_vars = np.var(predictions_matrix, axis=1)
    features["sample_pred_var_mean"] = _safe_float(np.mean(sample_vars))
    features["sample_pred_var_std"] = _safe_float(np.std(sample_vars))
    features["sample_pred_var_max"] = _safe_float(np.max(sample_vars))
    features["sample_pred_var_min"] = _safe_float(np.min(sample_vars))
    features["sample_pred_var_median"] = _safe_float(np.median(sample_vars))

    # Prediction entropy per sample
    if task_type == "classification":
        classes = np.unique(y_true)
        entropies = []
        for s in range(n_samples):
            preds_s = predictions_matrix[s, :].astype(int)
            _, counts = np.unique(preds_s, return_counts=True)
            probs = counts / counts.sum()
            entropies.append(float(sp_stats.entropy(probs)))
        entropies_arr = np.array(entropies)
        features["pred_entropy_mean"] = _safe_float(np.mean(entropies_arr))
        features["pred_entropy_std"] = _safe_float(np.std(entropies_arr))
        features["pred_entropy_max"] = _safe_float(np.max(entropies_arr))
    else:
        # Entropy of the models' predictions over the target's histogram bins
        # (regression analogue of the label entropy across models).
        edges = np.histogram_bin_edges(y_true, bins="auto")
        binned = np.digitize(predictions_matrix, edges)
        entropies_arr = np.array([
            sp_stats.entropy(np.unique(binned[s], return_counts=True)[1]) for s in range(n_samples)
        ])
        features["pred_entropy_mean"] = _safe_float(np.mean(entropies_arr))
        features["pred_entropy_std"] = _safe_float(np.std(entropies_arr))
        features["pred_entropy_max"] = _safe_float(np.max(entropies_arr))

    # Model agreement patterns
    if task_type == "classification":
        agreement_rates = []
        for s in range(n_samples):
            preds_s = predictions_matrix[s, :].astype(int)
            mode_val = sp_stats.mode(preds_s, keepdims=False).mode
            agreement_rates.append(float(np.mean(preds_s == mode_val)))
        agreement_arr = np.array(agreement_rates)
    else:
        # For regression: agreement = inverse of coefficient of variation
        agreement_arr = np.array([
            1.0 / (1.0 + np.std(predictions_matrix[s, :]) / (abs(np.mean(predictions_matrix[s, :])) + 1e-10))
            for s in range(n_samples)
        ])

    features["agreement_mean"] = _safe_float(np.mean(agreement_arr))
    features["agreement_std"] = _safe_float(np.std(agreement_arr))
    features["agreement_min"] = _safe_float(np.min(agreement_arr))
    features["agreement_max"] = _safe_float(np.max(agreement_arr))
    features["agreement_median"] = _safe_float(np.median(agreement_arr))

    # (Pairwise Pearson correlation of the predictions is not computed: for
    # regression it equals 1 - the pairwise diversity of the ensemble group, and for
    # multi-class labels a correlation of nominal class codes is not meaningful.)

    # Per-model prediction coverage (fraction of unique predictions)
    unique_ratios = [len(np.unique(predictions_matrix[:, i])) / n_samples for i in range(n_models)]
    features["unique_pred_ratio_mean"] = _safe_float(np.mean(unique_ratios))
    features["unique_pred_ratio_std"] = _safe_float(np.std(unique_ratios))
    features["unique_pred_ratio_max"] = _safe_float(np.max(unique_ratios))

    # Additional model statistics
    features["n_models"] = float(n_models)
    features["model_perf_p10"] = _safe_float(np.percentile(perfs_arr, 10))
    features["model_perf_p90"] = _safe_float(np.percentile(perfs_arr, 90))
    features["model_perf_cv"] = _safe_float(np.std(perfs_arr) / (np.mean(perfs_arr) + 1e-10))

    return features


def _extract_ensemble_characteristics(
    predictions_matrix: np.ndarray,
    y_true: np.ndarray,
    task_type: str,
    performance_metric: str | None = None,
    seed: int = 1985,
) -> dict:
    """Ensemble characteristics (40): diversity, subset-size performance, oracle and margins.

    Ensemble performance (majority vote / mean) uses the experiment's primary metric
    on the [0, 1] scale (see :func:`_score`).
    """
    from prunestack.metrics.diversity import (
        average_diversity,
        pairwise_diversity_matrix,
        disagreement_measure,
        correlation_coefficient,
    )
    n_samples, n_models = predictions_matrix.shape
    all_idx = list(range(n_models))
    features: dict[str, float] = {}
    metric = resolve_metric(performance_metric, task_type)

    def evaluate_ensemble_performance(y, preds, subset, task):  # noqa: ANN001 - local helper
        return evaluate_ensemble_score(y, preds, subset, task, performance_metric=metric)[0]

    # Full ensemble diversity
    div_mat = pairwise_diversity_matrix(predictions_matrix, all_idx, task_type, y_true)
    triu = div_mat[np.triu_indices(n_models, k=1)]
    features["diversity_mean"] = _safe_float(np.mean(triu))
    features["diversity_std"] = _safe_float(np.std(triu))
    features["diversity_max"] = _safe_float(np.max(triu))
    features["diversity_min"] = _safe_float(np.min(triu))
    features["diversity_median"] = _safe_float(np.median(triu))
    features["diversity_iqr"] = _safe_float(np.percentile(triu, 75) - np.percentile(triu, 25))
    features["diversity_skew"] = _safe_float(sp_stats.skew(triu))
    features["diversity_kurtosis"] = _safe_float(sp_stats.kurtosis(triu))

    # Full ensemble performance
    full_perf = evaluate_ensemble_performance(y_true, predictions_matrix, all_idx, task_type)
    features["full_ensemble_performance"] = _safe_float(full_perf)

    # Performance of random subsets of 1, 25 %, 50 % and 75 % of the pool
    rng = np.random.RandomState(seed)
    size_features = {
        "perf_at_small_subset": 1,
        "perf_at_quarter": max(1, n_models // 4),
        "perf_at_half": max(1, n_models // 2),
        "perf_at_three_quarter": max(1, 3 * n_models // 4),
    }
    for name, size in size_features.items():
        subset = rng.choice(n_models, size=size, replace=False).tolist()
        features[name] = _safe_float(
            evaluate_ensemble_performance(y_true, predictions_matrix, subset, task_type))
    features["perf_improvement_small_to_full"] = (
        features["full_ensemble_performance"] - features["perf_at_small_subset"])

    # Model ranking statistics
    perfs = []
    for i in range(n_models):
        p = evaluate_ensemble_performance(y_true, predictions_matrix, [i], task_type)
        perfs.append(p)
    perfs_arr = np.array(perfs)
    ranked = np.argsort(-perfs_arr)

    # Top-k ensemble performances
    for k_val in [3, 5, 10]:
        if k_val <= n_models:
            top_k = ranked[:k_val].tolist()
            features[f"top_{k_val}_ensemble_perf"] = _safe_float(
                evaluate_ensemble_performance(y_true, predictions_matrix, top_k, task_type))
            features[f"top_{k_val}_diversity"] = _safe_float(
                average_diversity(predictions_matrix, top_k, task_type, y_true))
        else:
            features[f"top_{k_val}_ensemble_perf"] = features["full_ensemble_performance"]
            features[f"top_{k_val}_diversity"] = features["diversity_mean"]

    # Bottom-k
    for k_val in [3, 5, 10]:
        if k_val <= n_models:
            bot_k = ranked[-k_val:].tolist()
            features[f"bottom_{k_val}_ensemble_perf"] = _safe_float(
                evaluate_ensemble_performance(y_true, predictions_matrix, bot_k, task_type))
        else:
            features[f"bottom_{k_val}_ensemble_perf"] = features["full_ensemble_performance"]

    # Diversity-performance correlation
    diversities_per_model = []
    for i in range(n_models):
        pair_divs = []
        for j in range(n_models):
            if i != j:
                if task_type == "classification":
                    pair_divs.append(disagreement_measure(predictions_matrix[:, i], predictions_matrix[:, j]))
                else:
                    pair_divs.append(1.0 - abs(correlation_coefficient(predictions_matrix[:, i], predictions_matrix[:, j])))
        diversities_per_model.append(np.mean(pair_divs) if pair_divs else 0.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(perfs_arr) > 1 and np.std(perfs_arr) > 0 and np.std(diversities_per_model) > 0:
            corr = np.corrcoef(perfs_arr, diversities_per_model)[0, 1]
            features["perf_diversity_correlation"] = _safe_float(corr)
        else:
            features["perf_diversity_correlation"] = 0.0

    # Error correlation
    if task_type == "classification":
        errors = (predictions_matrix.astype(int) != y_true[:, np.newaxis]).astype(float)
    else:
        errors = (predictions_matrix - y_true[:, np.newaxis]) ** 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if errors.shape[1] > 1:
            err_corr = np.corrcoef(errors, rowvar=False)
            err_triu = err_corr[np.triu_indices(n_models, k=1)]
            err_triu = err_triu[~np.isnan(err_triu)]
            features["error_correlation_mean"] = _safe_float(np.mean(err_triu)) if len(err_triu) > 0 else 0.0
            features["error_correlation_std"] = _safe_float(np.std(err_triu)) if len(err_triu) > 0 else 0.0
        else:
            features["error_correlation_mean"] = 0.0
            features["error_correlation_std"] = 0.0

    # Margin statistics (confidence in correct predictions)
    if task_type == "classification":
        # Margin = proportion of models predicting the correct class
        margins = []
        for s in range(n_samples):
            correct_count = np.sum(predictions_matrix[s, :].astype(int) == y_true[s])
            margins.append(correct_count / n_models)
        margins_arr = np.array(margins)
    else:
        # Margin = inverse of prediction std for each sample
        margins_arr = 1.0 / (np.std(predictions_matrix, axis=1) + 1e-10)

    features["margin_mean"] = _safe_float(np.mean(margins_arr))
    features["margin_std"] = _safe_float(np.std(margins_arr))
    features["margin_min"] = _safe_float(np.min(margins_arr))
    features["hard_sample_ratio"] = _safe_float(np.mean(margins_arr < np.median(margins_arr)))

    # Oracle performance (best model per sample)
    if task_type == "classification":
        oracle_correct = 0
        for s in range(n_samples):
            if any(predictions_matrix[s, i] == y_true[s] for i in range(n_models)):
                oracle_correct += 1
        features["oracle_performance"] = oracle_correct / n_samples
    else:
        oracle_errors = []
        for s in range(n_samples):
            errors_s = np.abs(predictions_matrix[s, :] - y_true[s])
            oracle_errors.append(np.min(errors_s))
        features["oracle_performance"] = _safe_float(1.0 - np.mean(oracle_errors) / (np.std(y_true) + 1e-10))

    features["oracle_vs_ensemble_gap"] = features["oracle_performance"] - features["full_ensemble_performance"]

    # Diversity percentiles
    features["diversity_p10"] = _safe_float(np.percentile(triu, 10))
    features["diversity_p90"] = _safe_float(np.percentile(triu, 90))

    # Ensemble gain over best individual
    best_individual = max(perfs)
    features["ensemble_gain_over_best"] = features["full_ensemble_performance"] - best_individual

    # Kappa-like diversity (for classification, proportion of models that agree with ensemble)
    if task_type == "classification":
        from prunestack.metrics.performance import ensemble_predictions_classification
        ens_pred = ensemble_predictions_classification(predictions_matrix, all_idx)
        model_ens_agreements = [np.mean(predictions_matrix[:, i].astype(int) == ens_pred) for i in range(n_models)]
        # The mean is not reported: for classification it equals agreement_mean.
        features["model_ensemble_agreement_std"] = _safe_float(np.std(model_ens_agreements))
    else:
        from prunestack.metrics.performance import ensemble_predictions_regression
        ens_pred = ensemble_predictions_regression(predictions_matrix, all_idx)
        # Correlation with ensemble
        corrs = [abs(float(np.corrcoef(predictions_matrix[:, i], ens_pred)[0, 1])) for i in range(n_models)]
        features["model_ensemble_agreement_std"] = _safe_float(np.std(corrs))

    # Diversity-to-performance ratio
    features["diversity_perf_ratio"] = features["diversity_mean"] / (features["full_ensemble_performance"] + 1e-10)

    # Additional ensemble features
    features["diversity_range"] = features["diversity_max"] - features["diversity_min"]
    features["n_weak_models"] = float(np.sum(perfs_arr < np.mean(perfs_arr) - np.std(perfs_arr)))
    features["n_strong_models"] = float(np.sum(perfs_arr > np.mean(perfs_arr) + np.std(perfs_arr)))

    return features


def _extract_optimization_results(
    optimization_results: dict | pd.DataFrame | None,
) -> dict:
    """Optimization characteristics over the alpha grid (best fitness, subset sizes, convergence).

    Without optimization results the same feature names are returned with value 0.0.
    """
    features: dict[str, float] = {}

    df = pd.DataFrame(optimization_results) if optimization_results is not None else pd.DataFrame()
    if df.empty:
        template = pd.DataFrame({
            "alpha": [0.0], "best_fitness": [0.0], "n_models_selected": [0],
            "execution_time": [0.0], "n_iterations": [0],
        })
        return dict.fromkeys(_extract_optimization_results(template), 0.0)

    # Best fitness statistics
    if "best_fitness" in df.columns:
        fitness_vals = df["best_fitness"].values
        features["opt_best_fitness_max"] = _safe_float(np.max(fitness_vals))
        features["opt_best_fitness_min"] = _safe_float(np.min(fitness_vals))
        features["opt_best_fitness_mean"] = _safe_float(np.mean(fitness_vals))
        features["opt_best_fitness_std"] = _safe_float(np.std(fitness_vals))
        features["opt_best_fitness_range"] = features["opt_best_fitness_max"] - features["opt_best_fitness_min"]
    else:
        for k in ["opt_best_fitness_max", "opt_best_fitness_min", "opt_best_fitness_mean",
                   "opt_best_fitness_std", "opt_best_fitness_range"]:
            features[k] = 0.0

    # Model selection patterns
    if "n_models_selected" in df.columns:
        sel_vals = df["n_models_selected"].values
        features["opt_n_selected_mean"] = _safe_float(np.mean(sel_vals))
        features["opt_n_selected_std"] = _safe_float(np.std(sel_vals))
        features["opt_n_selected_min"] = _safe_float(np.min(sel_vals))
        features["opt_n_selected_max"] = _safe_float(np.max(sel_vals))
        features["opt_n_selected_median"] = _safe_float(np.median(sel_vals))
    else:
        for k in ["opt_n_selected_mean", "opt_n_selected_std", "opt_n_selected_min",
                   "opt_n_selected_max", "opt_n_selected_median"]:
            features[k] = 0.0

    # Alpha sweep characteristics
    if "alpha" in df.columns:
        best_idx = df["best_fitness"].idxmax() if "best_fitness" in df.columns else 0
        features["opt_best_alpha"] = _safe_float(df.loc[best_idx, "alpha"])

        # Sensitivity to alpha
        alpha_groups = df.groupby("alpha")
        if "best_fitness" in df.columns:
            alpha_means = alpha_groups["best_fitness"].mean()
            features["opt_alpha_sensitivity"] = _safe_float(np.std(alpha_means))
        else:
            features["opt_alpha_sensitivity"] = 0.0

        features["opt_n_alpha_values"] = float(df["alpha"].nunique())
        features["opt_n_grid_points"] = float(len(df))
    else:
        for k in ["opt_best_alpha", "opt_alpha_sensitivity",
                   "opt_n_alpha_values", "opt_n_grid_points"]:
            features[k] = 0.0

    # Execution time statistics
    if "execution_time" in df.columns:
        times = df["execution_time"].values
        features["opt_time_mean"] = _safe_float(np.mean(times))
        features["opt_time_std"] = _safe_float(np.std(times))
        features["opt_time_total"] = _safe_float(np.sum(times))
    else:
        features["opt_time_mean"] = 0.0
        features["opt_time_std"] = 0.0
        features["opt_time_total"] = 0.0

    # Convergence characteristics
    if "n_iterations" in df.columns:
        iters = df["n_iterations"].values
        features["opt_iterations_mean"] = _safe_float(np.mean(iters))
        features["opt_iterations_std"] = _safe_float(np.std(iters))
        features["opt_iterations_max"] = _safe_float(np.max(iters))
        features["opt_early_stop_ratio"] = _safe_float(np.mean(iters < np.max(iters))) if np.max(iters) > 0 else 0.0
    else:
        features["opt_iterations_mean"] = 0.0
        features["opt_iterations_std"] = 0.0
        features["opt_iterations_max"] = 0.0
        features["opt_early_stop_ratio"] = 0.0

    # Performance landscape characteristics
    if "best_fitness" in df.columns:
        features["opt_fitness_cv"] = (features["opt_best_fitness_std"] /
                                       (features["opt_best_fitness_mean"] + 1e-10))
        # Rank correlation of fitness with alpha
        if "alpha" in df.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    corr = sp_stats.spearmanr(df["alpha"], df["best_fitness"])[0]
                    features["opt_fitness_alpha_corr"] = _safe_float(corr)
                except Exception:
                    features["opt_fitness_alpha_corr"] = 0.0
        else:
            features["opt_fitness_alpha_corr"] = 0.0

    else:
        features["opt_fitness_cv"] = 0.0
        features["opt_fitness_alpha_corr"] = 0.0

    # Pruning effectiveness
    if "n_models_selected" in df.columns and "best_fitness" in df.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                corr = sp_stats.spearmanr(df["n_models_selected"], df["best_fitness"])[0]
                features["opt_size_fitness_corr"] = _safe_float(corr)
            except Exception:
                features["opt_size_fitness_corr"] = 0.0
    else:
        features["opt_size_fitness_corr"] = 0.0

    return features


def extract_optimization_summary(optimization_results: dict | pd.DataFrame | None) -> dict[str, float]:
    """Summary of the EHO results over the alpha grid (24 values).

    Not a meta-feature: it is computed from the optimization outcome (target leakage
    for any meta-learner that predicts that outcome) and is saved separately.
    """
    return _extract_optimization_results(optimization_results)


def extract_meta_features(
    dataset_path: str | Path,
    base_models_df: pd.DataFrame | None,
    y_true: np.ndarray,
    X: np.ndarray,
    task_type: str = "classification",
    predictions_matrix: np.ndarray | None = None,
    performance_metric: str | None = None,
    folds: Folds | None = None,
    seed: int = 1985,
) -> dict[str, float]:
    """Extract the meta-features (123) of one dataset.

    Parameters
    ----------
    dataset_path : str or Path
        Dataset directory (used for logging only).
    base_models_df : DataFrame or None
        Out-of-fold base-model predictions (n_samples, n_models).
    y_true : ndarray
        Encoded labels / targets.
    X : ndarray
        Feature matrix.
    task_type : str
    predictions_matrix : ndarray or None
        Used instead of ``base_models_df`` when given.
    performance_metric : str or None
        Primary metric; ``None``/"auto" -> macro-F1 / RMSE (skill score).
    folds : list of (train_idx, val_idx) or None
        Folds for the landmarkers (the experiment's folds); ``None`` -> 5-fold.
    seed : int
        Seed of all stochastic meta-feature computations.

    Returns
    -------
    dict[str, float]
        Dataset (48), base-model (35) and ensemble (40) characteristics. The
        optimization summary is *not* included (see :func:`extract_optimization_summary`).
    """
    if predictions_matrix is not None:
        pred_mat = np.asarray(predictions_matrix, dtype=float)
    elif base_models_df is not None:
        pred_mat = base_models_df.to_numpy(dtype=float)
    else:
        raise ValueError("Either base_models_df or predictions_matrix must be provided")

    y_true = np.asarray(y_true)
    X = np.asarray(X, dtype=float)
    logger.info("Extracting meta-features for %s", dataset_path)
    groups = {
        "dataset": _extract_dataset_characteristics(X, y_true, task_type, performance_metric, folds, seed),
        "base_models": _extract_base_model_characteristics(pred_mat, y_true, task_type, performance_metric),
        "ensemble": _extract_ensemble_characteristics(pred_mat, y_true, task_type, performance_metric, seed),
    }
    all_features: dict[str, float] = {}
    for name, feats in groups.items():
        logger.info("  %-12s %d meta-features", name, len(feats))
        all_features.update(feats)

    nan_keys = [k for k, v in all_features.items() if isinstance(v, float) and np.isnan(v)]
    if nan_keys:
        logger.warning("Replacing %d NaN meta-features with 0.0: %s", len(nan_keys), nan_keys)
        all_features.update(dict.fromkeys(nan_keys, 0.0))

    logger.info("Total meta-features: %d", len(all_features))
    return all_features
