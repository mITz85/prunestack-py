# API reference (prunestack 2.0)

All public names are importable from their sub-package; the most common ones are
re-exported from `prunestack`.

## Configuration and experiment

| Name | Signature / purpose |
|---|---|
| `build_config` | `build_config(dataset_path, config_path=None, **overrides) -> ExperimentConfig` — defaults < YAML < keyword overrides |
| `ExperimentConfig` | dataclass of all options; `resolved_alpha_grid`, `resolved_optimization_seed`, `to_dict()` |
| `ExperimentRunner` | `ExperimentRunner(config, skip_optimization=False, skip_stacking=False, output_dir=None, strict=False)`; phases `load_data`, `train_base_models`, `evaluate_baseline`, `optimize_alpha_grid`, `train_stacking`, `compare_strategies`, `extract_meta_features`, `save_results`; `run_full_experiment() -> ExperimentResults` |
| `ExperimentResults` | `dataset_info`, `base_models_performance`, `optimization_results`, `stacking_results`, `strategy_comparison`, `meta_features`, `optimization_summary`, `execution_metadata` |
| `save_results` / `load_results` | `save_results(results, output_dir)`, `load_results(output_dir)` |

## Data (`prunestack.data`)

- `load_dataset(path, task_type="auto", n_folds=5, seed=42) -> DatasetBundle` — fields
  `X, y, metrics_df, folds, task_type, dataset_path, label_encoder, feature_encoders,
  feature_names, n_features, n_samples`. `n_folds` is reduced automatically for tiny
  datasets / rare classes.
- `detect_task_type(metrics_df, override="auto")`
- `run_all_validations(X, y, hyperparameters, config, task_type=None, strict=False)`,
  `validate_dataset`, `validate_hyperparameters`, `validate_config`, `ValidationResult`

## Models (`prunestack.models`)

- `load_hyperparameters(task_type, seed, config_path=None) -> dict`
- `load_stacking_config(task_type, config_path=None) -> (meta_learner_name, params)`
- `create_model_pool(n_features, hyperparameters, task_type, base_seed=42,
  max_models_per_algo=20, feature_subset_min_ratio=0.6, feature_subset_max_ratio=0.9) -> list[ModelConfig]`
- `generate_feature_subsets(n_features, n_variants=19, base_seed=42, algo_idx=0, min_ratio=0.6, max_ratio=0.9)`
- `train_base_models(X, y, model_pool, folds, task_type, performance_metric=None) -> (predictions_df, details_df)`
- `get_model_class(name, task_type)`, `list_algorithms(task_type=None)`

## Metrics (`prunestack.metrics`)

- `calculate_metric(y_true, y_pred, metric, task_type=None)`
- `evaluate_ensemble_performance(y_true, P, subset, task_type, performance_metric=None)`
- `normalize_performance_score(raw, metric, y_true)`, `is_lower_better(metric)`
- `average_diversity(P, subset, task_type, y_true=None, diversity_metric="auto")`,
  `pairwise_diversity_matrix(...)`, `resolve_diversity_metric(...)`
- `calculate_fitness(y_true, P, subset, alpha, task_type, performance_metric=None, diversity_metric="auto")`
- `evaluate_subset(...)` (same arguments) -> dict with performance, diversity, fitness

## Ensemble (`prunestack.ensemble`)

- `EnsembleEvaluator(y_true, P, task_type, performance_metric=None, diversity_metric="auto")`
  with `.evaluate(subset)` and `.evaluate_alpha_grid(subset, alpha_grid=None)`
- `evaluate_baseline(y_true, P, task_type, performance_metric=None, diversity_metric="auto") -> dict`
- `StackedEnsemble(base_predictions_matrix, y_true, task_type, cv_folds=5, meta_learner=None,
  meta_learner_params=None, seed=42)` with `.train()`, `.predict(subset=None)`,
  `.evaluate(subset_indices=None, performance_metric=None)`
- `build_meta_learner(name, task_type, n_classes=0, params=None, seed=42)`,
  `available_meta_learners(task_type)`, `META_LEARNERS`, `DEFAULT_META_LEARNER`
- `NNLSRegressor`, `NNLSMultiResponseClassifier(n_classes)`, `one_hot_labels`
- `compare_ensemble_strategies(y_true, P, optimized_subsets, task_type, cv_folds=5, seed=42,
  performance_metric=None, diversity_metric="auto", meta_learner=None, meta_learner_params=None)`
  -> DataFrame (`strategy_name, alpha, n_models, performance, diversity, fitness`)

## Optimization (`prunestack.optimization`)

- `EHOOptimizer(n_clans=5, clan_size=10, max_iterations=50, alpha=0.5, beta=0.1, early_stop_patience=10)`;
  `.optimize(objective_function, n_models, seed=42) -> OptimizationResult`
- `OptimizationResult`: `best_subset, best_fitness, convergence_history, n_iterations,
  execution_time, metadata`; `best_indices`, `n_selected`, `to_dict()`
- `OptimizerProtocol` — implement `optimize(objective_function, n_models, seed)` to plug in another optimizer
- `FitnessObjective(y_true, P, alpha, task_type, maximize=True, performance_metric=None, diversity_metric="auto")`
- `optimize_ensemble(y_true, P, alpha, task_type, optimizer, seed=42, ...)`
- `optimize_alpha_grid(y_true, P, alpha_grid, *, optimizer, task_type, seed=42, ...) -> DataFrame`
  (`alpha, best_fitness, best_subset_indices, n_models_selected, n_iterations, execution_time, convergence_history`)

```python
from prunestack.optimization import EHOOptimizer, FitnessObjective

objective = FitnessObjective(y, P, alpha=0.5, task_type="classification")
result = EHOOptimizer(max_iterations=50).optimize(objective, n_models=P.shape[1], seed=1)
print(result.best_indices, result.best_fitness)
```

## I/O (`prunestack.io`)

- `extract_meta_features(dataset_path, base_models_df, y_true, X, task_type="classification",
  predictions_matrix=None, performance_metric=None, folds=None, seed=1985) -> dict[str, float]`
  (123 features, see `meta_features.md`)
- `extract_optimization_summary(optimization_results) -> dict[str, float]` (24 `opt_*`
  values; analysis only, not meta-features)
- `generate_experiment_report(results, output_path, max_meta_features_shown=20)`
- `plot_optimization_convergence`, `plot_alpha_tradeoff`, `plot_strategy_comparison`,
  `plot_ensemble_size_vs_fitness` (require `matplotlib`)

## Utilities

- `setup_logging(*, verbose, quiet, log_file, level)`, `get_logger(name)`
- Exceptions: `PruneStackError` (base), `DataValidationError`, `ConfigurationError`,
  `OptimizationError`, `ModelTrainingError`, `InsufficientDataError`
