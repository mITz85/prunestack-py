# Algorithms

## Pipeline

`ExperimentRunner.run_full_experiment()` executes, for one dataset:

| # | Phase | Output |
|---|-------|--------|
| 1 | `load_data` – dataset, fixed hyperparameters, model pool, input validation | `base_models/model_pool.json` |
| 2 | `train_base_models` – k-fold out-of-fold predictions | `base_models/*.csv` |
| 3 | `evaluate_baseline` – full-pool performance and diversity | `experiment_summary.json` |
| 4 | `optimize_alpha_grid` – EHO subset selection per α | `optimization/*` |
| 5 | `train_stacking` – meta-learner on the full pool and on every `S_α` | `stacking/stacking_results.csv` |
| 6 | `compare_strategies` – vote/average vs. stacking, full vs. pruned | `comparison/strategy_comparison.csv` |
| 7 | `extract_meta_features` – 123 meta-features; optimization summary kept separate | `meta_features.json`, `optimization/optimization_summary.json` |
| 8 | `save_results` | summary, HTML report, plots |

## Model pool

Per algorithm: 1 original model (all features) + `max_models_per_algo − 1`
feature-subset variants. The number of features of a variant is drawn
uniformly from `[⌊0.6·p⌋, min(⌊0.9·p⌋, p − 1)]` and the features themselves
without replacement, with seed `seed + 1000·algo_index + variant_index`;
duplicate subsets are rejected and redrawn with the next index. The pool is
therefore identical across runs and optimization seeds.

| Classification | Regression | Non-default settings |
|----------------|------------|----------------------|
| DecisionTreeClassifier | DecisionTreeRegressor | – |
| KNeighborsClassifier | KNeighborsRegressor | – |
| LogisticRegression | Ridge | LR: `max_iter=1000` |
| MLPClassifier | MLPRegressor | `max_iter=1000` |
| HistGradientBoostingClassifier | HistGradientBoostingRegressor | `max_iter=300` |
| RandomForestClassifier | ElasticNet | RF: `n_estimators=100` (default); EN: `max_iter=2000` |

Hyperparameters are fixed for all datasets and all models of an algorithm;
only iteration limits are raised to ensure convergence.

### Out-of-fold predictions

For every model and fold, median imputation and standardisation are fitted on
the training part only, the model is trained on the training part and predicts
the held-out part. Classification pools store encoded class labels
(0 … K−1), regression pools real values. The resulting
`n_samples × n_models` matrix is the input of the optimization, diversity and
meta-feature phases. For classification the out-of-fold class probabilities
(`predict_proba`, `n_samples × n_models × K`) are stored as well
(`base_models_probabilities.csv`) and are the default level-0 inputs of stacking.

## Elephant Herding Optimization

Binary EHO (Wang, Deb & Coelho, 2015). Every elephant holds a continuous
position `x ∈ ℝ^M` (M = pool size), initialised from N(0, 1); the subset is
`S = {m : x_m ≥ 0}` (sigmoid ≥ 0.5). Empty subsets are repaired by selecting
one random model.

In iteration `t = 1 … T`, for every clan `c` (elephants sorted by fitness, the
best one is the matriarch):

* **clan update** – `x ← x + α_t · r ⊙ (x_matriarch − x)`, `r ~ U(0,1)^M`,
  with the step growing linearly from exploration to exploitation:
  `α_t = α_EHO + (1 − α_EHO) · t / T`;
* **matriarch update** – `x_matriarch ← x_matriarch + β_EHO · r ⊙ (x_centre − x_matriarch)`;
* **separation** – the worst elephant of the clan is re-initialised from N(0, 1).

The global best subset is tracked; the search stops after
`eho_max_iterations` or after `eho_early_stop_patience` iterations without
improvement. Fitness values are cached per subset.

Default population: 5 clans × 10 elephants, 50 iterations, α_EHO = 0.5,
β_EHO = 0.1, patience 10.

Any object implementing `prunestack.optimization.protocol.OptimizerProtocol`
(`optimize(objective_function, n_models, seed) -> OptimizationResult`) can be
used with `optimize_ensemble` / `optimize_alpha_grid`.

## Stacking

Level-0 inputs are the out-of-fold base-model outputs: for classification by
default the class probabilities `p_{m,k}` (`stacking_inputs: probabilities`,
Ting & Witten, 1999), optionally the predicted labels (`stacking_inputs: labels`);
for regression the predicted values. Pruning, diversity and the meta-features
always use the labels, so the choice affects the stacking scores only. The
level-1 meta-learner is scored with an inner stratified (classification) or
plain (regression) k-fold, so every stacking score is out-of-fold as well.

### `nnls_mlr` – classification default

Multi-response linear regression with non-negative coefficients
(Ting & Witten, 1999). For each class `k`:

```
[y = k]  ≈  Σ_m  w_{k,m} · p_{m,k},      w_{k,m} ≥ 0      (scipy.optimize.nnls)
```

The predicted class is `argmax_k Σ_m w_{k,m} · p_{m,k}`; ties are broken by
the plain vote count. With labels, `p_{m,k}` is replaced by `[ŷ_m = k]` and the
combiner becomes a learned, class-specific weighted vote. Zero weights prune
models implicitly. A class whose weights are all zero falls back to equal weights.

### `nnls` – regression default

Non-negative least squares without intercept (Breiman, 1996):
`ŷ = Σ_m w_m · ŷ_m`, `w_m ≥ 0`.

### Why not logistic regression by default

On hard class labels an unconstrained linear model can assign negative weights,
i.e. "trust model m less when it votes for k", which over-fits correlated pool
members and is hard to interpret. Non-negative linear combiners are the
established choice for stacking (Breiman, 1996; Ting & Witten, 1999), need no
tuning and keep the weights interpretable as model contributions.
`logistic` and the other meta-learners remain available for sensitivity
analyses (`--stacking-method`, `examples/stacking_only.py`).

| Name | Classification | Regression |
|------|----------------|------------|
| `nnls_mlr` | ✓ default | – |
| `nnls` | – | ✓ default |
| `ridge` | `RidgeClassifier` | `Ridge` |
| `logistic` | `LogisticRegression` | – |
| `categorical_nb` | `CategoricalNB` | – |
| `random_forest` | `RandomForestClassifier` | `RandomForestRegressor` |
| `hist_gradient_boosting` | `HistGradientBoostingClassifier` | `HistGradientBoostingRegressor` |

For classification the scikit-learn meta-learners (except `categorical_nb`,
which uses the labels directly) receive `n_models · n_classes` columns: the
class probabilities, or the one-hot encoded labels with `stacking_inputs: labels`. Parameters for them can be set in
`stacking.<task>.params` of the hyperparameters JSON.

## Strategy comparison

For every α:

| Strategy | Models | Combiner |
|----------|--------|----------|
| `full_ensemble` | whole pool | majority vote / mean |
| `optimized_ensemble` | `S_α` | majority vote / mean |
| `full_stacked` | whole pool | stacking meta-learner |
| `optimized_stacked` | `S_α` | stacking meta-learner |
