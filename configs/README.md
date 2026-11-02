# Base-learner hyperparameter configuration

`hyperparameters.json` defines the **fixed hyperparameters** of every base learner
in the prunestack model pool. No per-dataset hyperparameter optimisation is
performed: the same parameters are used for the original model (all features)
and for every feature-subset variant of an algorithm, on every dataset. The only
source of within-algorithm variation is the random feature subset.

A copy of this file is bundled inside the Python package
(`src/prunestack/configs/hyperparameters.json`) so that a `pip install` works
without the repository checkout. Keep the two copies identical (a unit test
checks this).

## Format

```json
{
  "_description": ["..."],
  "_version": "2.0",
  "_sklearn_version": ">=1.9",
  "_notes": ["..."],
  "classification": {
    "DecisionTreeClassifier": {},
    "LogisticRegression": {"max_iter": 1000},
    "...": {}
  },
  "regression": {
    "Ridge": {},
    "ElasticNet": {"max_iter": 2000},
    "...": {}
  },
  "stacking": {
    "classification": {"meta_learner": "nnls_mlr", "params": {}},
    "regression": {"meta_learner": "nnls", "params": {}}
  }
}
```

- **Metadata keys** start with `_` (`_description`, `_version`, `_sklearn_version`,
  `_notes`). They are documentation only and are ignored by the loader (also inside
  a task section).
- **`classification` / `regression`** map each algorithm name to keyword arguments
  of the corresponding scikit-learn estimator. An empty object `{}` means pure
  scikit-learn defaults. The names must match **exactly** the algorithm registry
  (`prunestack.models.registry.CLASSIFICATION_ALGORITHMS` /
  `REGRESSION_ALGORITHMS`); a missing or unknown name raises `ValueError`.
- **`stacking`** selects the meta-learner per task (`meta_learner`) and its keyword
  arguments (`params`). Defaults: `nnls_mlr` (non-negative multi-response linear
  regression, Ting & Witten 1999) for classification and `nnls` (non-negative least
  squares, Breiman 1996) for regression. Alternatives: `ridge`, `logistic`,
  `categorical_nb` (classification), `random_forest`, `hist_gradient_boosting`.
  The `--stacking-method` CLI flag overrides the file.

## `random_state` injection

`random_state` is **not** stored in the file. At runtime the experiment seed is
injected as `random_state` for every estimator that accepts it (checked via
`estimator().get_params()`), e.g. DecisionTree, LogisticRegression, MLP,
HistGradientBoosting, RandomForest, Ridge and ElasticNet, but not kNN.

## Using a custom file

Resolution order:

1. an explicit path: `--hyperparameters PATH` on the command line, or
   `hyperparameters_path: PATH` in the YAML experiment config;
2. the repository-level `configs/hyperparameters.json`;
3. the copy bundled with the installed package.

```bash
prunestack --dataset-path data/my_dataset \
    --hyperparameters my_hyperparameters.json
```

```python
from prunestack.models.hyperparameters import load_hyperparameters, load_stacking_config

params = load_hyperparameters("classification", seed=42)            # default file
params = load_hyperparameters("regression", seed=42, config_path="my.json")
name, meta_params = load_stacking_config("classification")
```

## Methods statement

> All base learners use scikit-learn default hyperparameters, with minor
> adjustments to iteration limits to ensure convergence; the full configuration is
> provided in configs/hyperparameters.json in the public repository.
