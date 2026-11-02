# Quick start

## 1. Prepare a dataset

```
my_dataset/
├── X_test.csv     # index column + feature columns
├── y_test.csv     # index column + target column
└── metrics.csv    # index = algorithm names, columns = metric names (task detection only)
```

Example `metrics.csv` for a classification dataset:

```csv
,accuracy,f1
DecisionTreeClassifier,0.81,0.79
```

## 2. Run

```bash
prunestack --dataset-path my_dataset
```

This runs the full protocol: 120 base models, 5-fold out-of-fold predictions,
EHO pruning for α ∈ {0.0, 0.1, …, 1.0}, stacking, strategy comparison and
meta-feature extraction. For a quick check use a reduced setting:

```bash
prunestack --dataset-path my_dataset \
    --max-models-per-algo 3 --alpha-grid 0,0.5,1 --eho-max-iterations 5
```

## 3. Read the results

| File | What to look at |
|------|-----------------|
| `results/comparison/strategy_comparison.csv` | `full_ensemble`, `optimized_ensemble`, `full_stacked`, `optimized_stacked` for every α |
| `results/optimization/optimization_results.csv` | selected subset `S_α`, its size and fitness |
| `results/stacking/stacking_results.csv` | meta-learner score on the full pool and on every `S_α` |
| `results/meta_features.json` | 123 meta-features of the dataset (see `meta_features.md`) |
| `results/experiment_report.html` | summary tables and plots |

`performance` in the comparison table is the raw metric (macro-F1 or RMSE);
`fitness` uses the [0, 1]-normalised performance.

## 4. From Python

```python
from prunestack.config import build_config
from prunestack.experiment.runner import ExperimentRunner
from prunestack.utils.logging import setup_logging

setup_logging(quiet=True)
config = build_config("my_dataset", alpha_grid=[0.0, 0.5, 1.0])
results = ExperimentRunner(config, output_dir="my_results").run_full_experiment()

print(results.optimization_results[["alpha", "n_models_selected", "best_fitness"]])
print(results.strategy_comparison)
```

Previously saved results can be reloaded with
`prunestack.io.results.load_results("my_results")`.
