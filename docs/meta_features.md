# Meta-Features (123)

`extract_meta_features()` describes one dataset by **123 meta-features**, the
inputs of the meta-learning layer. All of them can be computed *before* pruning
is run, from the data and the out-of-fold prediction matrix of the base-model
pool. The feature names are identical for classification and regression; every
value is a float. Values that cannot be computed (degenerate inputs) are
reported as `0.0` and logged.

| Group | Count | Input |
|-------|------:|-------|
| Dataset characteristics | 48 | `X`, `y`, experiment folds |
| Base-model characteristics | 35 | out-of-fold prediction matrix, `y` |
| Ensemble characteristics | 40 | out-of-fold prediction matrix, `y` |
| **Total** | **123** | written to `meta_features.json` |

The EHO **optimization summary** (24 values) is *not* a meta-feature; see
[below](#optimization-summary-not-a-meta-feature).

## Metrics and resampling

* **Performance-based meta-features** (landmarkers, nearest-neighbour error,
  `model_perf_*`, ensemble performances) use the experiment's primary metric on
  the [0, 1] fitness scale of `normalize_performance_score`, i.e. the same
  scale as `P(S)` in the fitness function:
  * classification – **macro-F1**;
  * regression – **RMSE skill score** `1 − RMSE / sd(y)` (clipped to [0, 1]);
    scale-free, hence comparable across datasets.
* **Landmarkers and the 1-NN error** are cross-validated on the **experiment's
  folds** (default 5-fold, stratified for classification) and all stochastic
  computations use the experiment seed (default 1985).

## 1. Dataset characteristics (48)

| Category | Features |
|----------|----------|
| Simple / statistical (12) | `n_samples`, `n_features`, `n_classes` (0 for regression), `class_imbalance_ratio` (minority / majority; 0 for regression), `feature_to_sample_ratio`, `mean_feature_correlation`, `max_feature_correlation` (absolute Pearson), `mean_feature_std`, `mean_feature_skewness`, `mean_feature_kurtosis`, `missing_value_ratio`, `categorical_feature_ratio` (< 10 distinct values) |
| Information-theoretic (9) | `feature_entropy_mean`, `label_entropy` (binned for regression), `mutual_information_mean/max/std` (feature–target, k-NN estimator), `feature_mutual_information_mean` (≤ 50 random feature pairs), `joint_entropy_mean`, `normalized_mutual_information_mean`, `information_gain_ratio_mean` |
| Complexity / geometry (9) | `fisher_discriminant_ratio`, `class_separability_score` (mean centroid distance), `volume_of_overlap` (classification only, else 0), `feature_efficiency` (share of features with MI > 0.01), `dimensionality_curse_indicator` (p / ln(n+1)), `nearest_neighbor_error_rate` (1 − CV score of 1-NN), `decision_tree_depth`, `pca_explained_variance_95` (components for 95 % variance), `intrinsic_dimensionality_estimate` (MLE, k = 10) |
| Landmarking (6) | `landmark_<name>_score` – CV score of `naive_bayes` (GaussianNB; 0 for regression), `decision_stump_depth3`, `linear_svm`, `knn3`, `random_forest10`, `linear_model` (logistic regression / ridge) |
| Structural (12) | `log_n_samples`, `log_n_features`, `sample_feature_product`, `sparsity_ratio`, `mean/std/max/min_feature_range`, `outlier_ratio` (\|z\| > 3), `target_mean`, `target_std`, `target_skewness` |

Landmarkers are dataset descriptors only; they are not part of the model pool
and use default settings (no tuning).

## 2. Base-model characteristics (35)

| Category | Features |
|----------|----------|
| Individual performance (14) | `model_perf_mean/std/max/min/median/range/iqr/skew/kurtosis/p10/p90/cv`, `n_models_above_mean`, `n_models` |
| Prediction statistics (10) | `pred_mean_of_means`, `pred_std_of_means`, `pred_mean_of_stds`, `pred_std_of_stds`, `pred_range_mean`, `sample_pred_var_mean/std/max/min/median` |
| Prediction entropy (3) | `pred_entropy_mean/std/max` – per-sample entropy of the models' predictions (classes; regression: target histogram bins) |
| Agreement (8) | `agreement_mean/std/min/max/median` (share of models voting for the per-sample mode; regression: 1 / (1 + CV)), `unique_pred_ratio_mean/std/max` |

## 3. Ensemble characteristics (40)

| Category | Features |
|----------|----------|
| Pairwise diversity (13) | `diversity_mean/std/max/min/median/iqr/skew/kurtosis/p10/p90/range` (disagreement / ambiguity, as in the fitness), `diversity_perf_ratio`, `perf_diversity_correlation` |
| Ensemble performance (11) | `full_ensemble_performance`, `perf_at_small_subset/quarter/half/three_quarter` (random subsets of 1, 25 %, 50 %, 75 % of the pool), `perf_improvement_small_to_full`, `ensemble_gain_over_best`, `n_weak_models`, `n_strong_models`, `oracle_performance`, `oracle_vs_ensemble_gap` |
| Top-/bottom-k (9) | `top_{3,5,10}_ensemble_perf`, `top_{3,5,10}_diversity`, `bottom_{3,5,10}_ensemble_perf` |
| Errors and margins (7) | `error_correlation_mean/std`, `margin_mean/std/min` (classification: share of correct votes; regression: 1 / sd of predictions), `hard_sample_ratio`, `model_ensemble_agreement_std` |

## Removed duplicate features

Earlier versions reported features that were exact copies, complements or
affine functions of another feature. They carry no additional information,
inflate the apparent dimensionality and distort feature-importance analyses
(the importance of one quantity is split across its copies), so they were removed:

| Removed | Reason |
|---------|--------|
| `class_entropy` | alias of `label_entropy` |
| `numerical_feature_ratio` | `1 − categorical_feature_ratio` |
| `full_ensemble_diversity` | copy of `diversity_mean` |
| `perf_at_full` | copy of `full_ensemble_performance` |
| `perf_spread` | copy of `perf_improvement_small_to_full` |
| `effective_ensemble_size` | `n_models · diversity_mean` (pool size is constant) |
| `error_rate_mean/std/min` | `1 − model_perf_*` (complements of individual performance) |
| `model_ensemble_agreement_mean` | equals `agreement_mean` for classification |
| `model_correlation_mean/std/max/min/median` | regression: `1 −` pairwise diversity (ambiguity group); classification: Pearson correlation of nominal class codes is not meaningful |

Two definitions were corrected because they duplicated other features:
the regression `landmark_naive_bayes_score` used ridge regression (identical to
`landmark_linear_model_score`) and is now 0.0, and the regression
`pred_entropy_*` were the per-sample prediction variances (identical to
`sample_pred_var_*`) and are now real entropies over target bins.

## Optimization summary (not a meta-feature)

`extract_optimization_summary(optimization_df)` returns 24 `opt_*` values
(best fitness, subset sizes, best α, α-sensitivity, run time, iterations and
correlations over the α grid). They are computed **from the optimization
outcome itself** – e.g. `opt_best_alpha` is the best α – so using them as inputs
of a meta-learner that predicts that outcome would be target leakage and
would inflate its accuracy. They are therefore written separately to
`optimization/optimization_summary.json` (and `ExperimentResults.optimization_summary`)
for descriptive analysis only.

## Usage

```python
from prunestack.io.meta_features import extract_meta_features, extract_optimization_summary

meta = extract_meta_features(
    dataset_path="./my_dataset",
    base_models_df=None,
    y_true=y,
    X=X,
    task_type="classification",
    predictions_matrix=pred_matrix,      # out-of-fold, n_samples × n_models
    performance_metric="f1",             # macro-F1; None -> macro-F1 / RMSE
    folds=folds,                         # experiment folds; None -> 5-fold
    seed=1985,
)
assert len(meta) == 123

summary = extract_optimization_summary(opt_df)   # analysis only
```
