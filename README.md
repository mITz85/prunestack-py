# PruneStack

**Meta-feature guided performance–diversity pruning of heterogeneous ensembles**

[![Python 3.15](https://img.shields.io/badge/python-3.15-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

PruneStack is the reference implementation of the experimental pipeline in
*Learning to Prune: Meta-Feature Guided Performance–Diversity Optimization for
Heterogeneous Ensembles* (Domladovac, 2026). For one dataset it

1. builds a heterogeneous pool of base models with **fixed hyperparameters**
   (6 algorithms × 20 models = 120 models),
2. collects their out-of-fold predictions with k-fold cross-validation,
3. prunes the pool with **Elephant Herding Optimization (EHO)** for every
   trade-off weight α of the fitness function,
4. combines the full and the pruned pools by **majority vote / averaging** and
   by **stacked generalization**, and
5. extracts **123 meta-features** describing the dataset, the pool and the
   ensemble, for the meta-learning layer (the optimization outcome is
   summarised separately, so that it never leaks into the meta-features).

---

## Method in brief

### Fitness function

```
F(S) = α · P̃(S) + (1 − α) · D̃(S),      α ∈ {0.0, 0.1, …, 1.0}
P̃(S) = (P(S) − P_lo) / (P_hi − P_lo),   D̃(S) = (D(S) − D_lo) / (D_hi − D_lo)
```

| Symbol | Classification | Regression |
|--------|----------------|------------|
| `P(S)` | macro-F1 of the majority vote of `S` | RMSE skill score `1 − RMSE(S) / std(y)` of the mean of `S`, clipped to [0, 1] |
| `D(S)` | mean pairwise disagreement | normalised ambiguity `Ā / Ē` (Krogh & Vedelsby, 1995) |

P and D lie in [0, 1] but in very different ranges (e.g. macro-F1 ≈ 0.75,
disagreement ≈ 0.25). Before mixing, both are min–max normalised with bounds
`[lo, hi]` = 1st/99th percentile over 500 random reference subsets drawn once per
dataset (seed 1985), so α weights comparable quantities: α = 1 optimizes
performance only, α = 0 diversity only. The map is affine and not clipped, so it
never changes the ranking of subsets for a fixed α. `fitness_normalization: none`
restores the raw fitness `α·P + (1 − α)·D`.

### Optimizer

Binary EHO (Wang, Deb & Coelho, 2015) with 10 clans × 10 elephants, up to 100
iterations and patience 25. The standard operator set uses a size-diverse
initialisation (per-elephant inclusion probability p ~ U(0.05, 0.5)), the
original matriarch (`β_EHO · clan centre`) and separating (uniform re-sampling)
operators and bit-flip mutation (1/M). `eho_variant: legacy` reproduces the
operators of PruneStack ≤ 2.0.0. An equal-budget benchmark against random search,
greedy forward selection, hill climbing, simulated annealing, a genetic algorithm
and binary PSO is in [`scripts/benchmark_optimizers.py`](scripts/benchmark_optimizers.py);
design and rationale: [`docs/manuscript_decisions.md`](docs/manuscript_decisions.md).

### Model pool

For every algorithm the pool contains **1 original model** trained on all
features and **19 feature-subset variants**, each trained on a random 60–90 %
of the features. All models of one algorithm share the same fixed
hyperparameters (scikit-learn defaults with raised iteration limits, see
[`configs/hyperparameters.json`](configs/hyperparameters.json)); they differ
only in the features they see. No per-dataset tuning is performed.

| Classification | Regression |
|----------------|------------|
| DecisionTreeClassifier | DecisionTreeRegressor |
| KNeighborsClassifier | KNeighborsRegressor |
| LogisticRegression | Ridge |
| MLPClassifier | MLPRegressor |
| HistGradientBoostingClassifier | HistGradientBoostingRegressor |
| RandomForestClassifier | ElasticNet |

Preprocessing (median imputation + standardisation) is fitted inside every
training fold, so the out-of-fold predictions are leakage-free.

### Stacking meta-learners

The stacking layer is trained on the out-of-fold base-model predictions and is
itself scored with an inner k-fold cross-validation.

| Name | Task | Description |
|------|------|-------------|
| `nnls_mlr` *(default)* | classification | Multi-response linear regression with non-negative weights (Ting & Witten, 1999): for each class `k`, NNLS of `[y = k]` on the base-model indicators `[ŷ_m = k]`; arg-max of the responses. A learned, class-specific weighted vote. |
| `nnls` *(default)* | regression | Non-negative least-squares combination without intercept (Breiman, 1996): `ŷ = Σ w_m · ŷ_m`, `w_m ≥ 0`. |
| `ridge` | both | `Ridge` / `RidgeClassifier` |
| `logistic` | classification | Multinomial `LogisticRegression` |
| `categorical_nb` | classification | `CategoricalNB` on the predicted labels |
| `random_forest` | both | `RandomForestClassifier` / `RandomForestRegressor` |
| `hist_gradient_boosting` | both | `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` |

The non-negative linear combiners are the defaults because they are the
established stacking combiners for each task, have no hyperparameters to
tune, and their weights are directly interpretable as model contributions.

---

## Installation

PruneStack requires **Python 3.15**.

```bash
git clone https://github.com/mITz85/prunestack-py.git
cd prunestack-py
python3.15 -m venv .venv
source .venv/bin/activate
pip install -e ".[plots]"          # add ,dev for the test suite
```

Exact versions used for the manuscript are pinned in
[`requirements-lock.txt`](requirements-lock.txt):

```bash
pip install -r requirements-lock.txt
pip install -e . --no-deps
```

---

## Quick start

```bash
prunestack --dataset-path data/my_dataset
```

Results are written to `data/my_dataset/results/`. A quick, reduced run:

```bash
prunestack --dataset-path data/my_dataset \
    --max-models-per-algo 5 --alpha-grid 0,0.5,1 --eho-max-iterations 10
```

From Python:

```python
from prunestack.config import build_config
from prunestack.experiment.runner import ExperimentRunner

config = build_config("data/my_dataset", seed=1985)
results = ExperimentRunner(config).run_full_experiment()
print(results.strategy_comparison)
```

Thirty optimization runs with a single training of the base models, and
parallel training of the pool (see [docs/configuration.md](docs/configuration.md#parallel-execution)):

```bash
prunestack --dataset-path data/my_dataset --seed 1985 --n-runs 30 --n-jobs -1
scripts/run_all.sh data 8          # all datasets of data/, 8 processes in parallel
DATASETS_FILE=configs/datasets_final.txt scripts/run_all.sh data 8   # the 21 manuscript datasets
```

See [`examples/`](examples/) for batch processing, custom optimization
settings, meta-learner comparison, meta-feature export and plotting.

---

## Input format

A dataset is a directory with three CSV files, each with an index column:

```
my_dataset/
├── X_test.csv     # features; numeric or categorical (categorical columns are label-encoded)
├── y_test.csv     # target in the first column
└── metrics.csv    # rows = algorithms, columns = metric names; used only to detect the task type
```

The task type is detected from the column names of `metrics.csv`
(`accuracy`, `f1`, `precision`, `recall`, `kappa` → classification;
`mae`, `mse`, `rmse`, `r2` → regression), with a fallback on the target values.
Use `--task-type` to set it explicitly.

---

## Configuration

Precedence: command-line arguments > YAML file (`--config`) > defaults.
The defaults reproduce the protocol of the manuscript.

| Option | Default | Description |
|--------|---------|-------------|
| `seed` | 1985 | Seed for CV folds, feature subsets and models |
| `n_folds` | 5 | Cross-validation folds |
| `task_type` | `auto` | `auto`, `classification`, `regression` |
| `max_models_per_algo` | 20 | 1 original model + 19 feature-subset variants |
| `feature_subset_min_ratio` | 0.6 | Minimum share of features in a variant |
| `feature_subset_max_ratio` | 0.9 | Maximum share of features in a variant |
| `performance_metric` | `auto` | `P(S)`: `f1` (macro) / `rmse`; also `accuracy`, `mae`, `mse`, `r2` |
| `diversity_metric` | `auto` | `D(S)`: `disagreement` / `ambiguity`; also `q_statistic`, `double_fault`, `correlation`, `rmse_difference` |
| `alpha_grid` | 0.0, 0.1, …, 1.0 | Trade-off weights α |
| `hyperparameters_path` | bundled JSON | Fixed base-model hyperparameters and stacking settings |
| `stacking_method` | from JSON | Overrides the meta-learner of the JSON file |
| `fitness_normalization` | `minmax` | `minmax` (normalise P and D before mixing) or `none` |
| `fitness_reference_subsets` | 500 | Random subsets used to estimate the normalisation bounds |
| `optimizer` | `eho` | Subset optimizer |
| `eho_variant` | `standard` | EHO operator set; `legacy` = PruneStack ≤ 2.0.0 |
| `eho_n_clans` | 10 | Number of clans |
| `eho_clan_size` | 10 | Elephants per clan |
| `eho_max_iterations` | 100 | Maximum iterations |
| `eho_alpha` | 0.5 | Matriarch influence α_EHO |
| `eho_beta` | 0.1 | Clan-centre factor β_EHO |
| `eho_early_stop_patience` | 25 | Iterations without improvement before stopping (0 = off) |
| `eho_init_inclusion_min` / `_max` | 0.05 / 0.5 | Range of the per-elephant inclusion probability at initialisation |
| `eho_mutation_rate` | 1/M | Bit-flip probability per bit |
| `eho_max_evaluations` | unlimited | Fitness-evaluation budget |
| `optimization_seed` | = `seed` | Seed of the optimizer (repeated runs use the run number) |

Example YAML:

```yaml
seed: 1985
alpha_grid: [0.0, 0.25, 0.5, 0.75, 1.0]
stacking_method: hist_gradient_boosting
eho_max_iterations: 100
```

Run `prunestack --help` for the full list of command-line options, including
`--output-dir`, `--skip-optimization`, `--skip-stacking`, `--strict`,
`--verbose` and `--quiet`. Details: [`docs/configuration.md`](docs/configuration.md).

---

## Output

```
results/
├── base_models/
│   ├── base_models_predictions.csv   # out-of-fold predictions (n_samples × n_models)
│   ├── base_models_probabilities.csv # classification: out-of-fold predict_proba ({model}__p{k})
│   ├── base_models_performance.csv   # per model and fold: metric, features used, timing
│   └── model_pool.json               # algorithm, feature indices and hyperparameters of every model
├── optimization/
│   ├── optimization_results.csv      # per α: best fitness, selected subset, iterations, time
│   ├── optimization_convergence.json
│   ├── fitness_normalization.json    # P/D bounds of the normalised fitness
│   └── optimization_summary.json     # 24 opt_* values (analysis only, not meta-features)
├── stacking/stacking_results.csv     # meta-learner score on the full pool and on every S_α
├── comparison/strategy_comparison.csv  # full/pruned × vote/stacking for every α
├── meta_features.json                # 123 meta-features
├── experiment_summary.json           # dataset info, configuration, package version, phase timings
├── experiment_report.html
├── plots/                            # alpha_tradeoff, size_vs_fitness, convergence, strategy_comparison
└── experiment.log
```

With `--n-runs N` the shared `base_models/` stays at the top level and every run
writes its own `run_01/ ... run_N/` directory with the structure above.

Model names are `<Algorithm>_original` and `<Algorithm>_subset_01` … `_subset_19`.

---

## Reproducibility

* All randomness derives from `seed`; the feature subset of variant `v` of
  algorithm `a` uses seed `seed + 1000·a + v`.
* `random_state` is injected into every estimator that accepts it.
* `experiment_summary.json` stores the full resolved configuration and the
  package version of every run.
* Results do not depend on `--n-jobs` or on how many processes run in parallel.
* Dependency versions are pinned in `requirements-lock.txt`.

## Reproducing the manuscript

The folder [`paper/`](paper/README.md) contains the per-dataset result files of the
experiment reported in the manuscript (21 datasets, 30 runs each) and the scripts that
recompute every table, statistic and figure from them. See [`paper/README.md`](paper/README.md).

---

## Documentation

| Document | Content |
|----------|---------|
| [installation](docs/installation.md) | Environment setup and verification |
| [quickstart](docs/quickstart.md) | First run and reading the results |
| [configuration](docs/configuration.md) | All options, YAML, hyperparameters JSON |
| [algorithms](docs/algorithms.md) | Pipeline, model pool, EHO, stacking |
| [metrics](docs/metrics.md) | Performance and diversity measures, fitness |
| [manuscript_decisions](docs/manuscript_decisions.md) | Methodological decisions for the manuscript: datasets, fitness normalisation, EHO, optimizer benchmark |
| [meta_features](docs/meta_features.md) | The 123 meta-features |
| [api_reference](docs/api_reference.md) | Python API |
| [faq](docs/faq.md), [troubleshooting](docs/troubleshooting.md) | Common questions and errors |

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

---

## Citation

If you use PruneStack, please cite the article and the software
(see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{domladovac_prunestack_2026,
  author  = {Domladovac, Marko},
  title   = {{PruneStack}: Meta-Feature Guided Performance--Diversity Pruning of Heterogeneous Ensembles},
  year    = {2026},
  version = {2.2.0},
  doi     = {10.5281/zenodo.XXXXXXX},
  url     = {https://github.com/mITz85/prunestack-py}
}
```

## References

* Breiman, L. (1996). Stacked regressions. *Machine Learning*, 24(1), 49–64.
* Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. *NIPS 7*, 231–238.
* Kuncheva, L. I., & Whitaker, C. J. (2003). Measures of diversity in classifier ensembles. *Machine Learning*, 51(2), 181–207.
* Ting, K. M., & Witten, I. H. (1999). Issues in stacked generalization. *Journal of Artificial Intelligence Research*, 10, 271–289.
* Wang, G.-G., Deb, S., & Coelho, L. d. S. (2015). Elephant herding optimization. *ISCBI 2015*, 1–5.
* Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.

## License

MIT — see [LICENSE](LICENSE).
