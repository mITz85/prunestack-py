# Configuration

A configuration is built from three sources, highest priority first:

1. command-line arguments (or keyword arguments of `build_config`),
2. a YAML file passed with `--config`,
3. `prunestack.config.DEFAULTS`, which reproduce the manuscript protocol.

Unknown keys in the YAML file raise an error.

## Options

| Key | CLI flag | Default | Description |
|-----|----------|---------|-------------|
| `seed` | `--seed` | 1985 | Seed for CV folds, feature subsets and estimators |
| `protocol` | `--protocol` | `nested` | `nested`: selection on inner out-of-fold predictions of the outer training part, evaluation on the outer held-out part (manuscript); `in_sample`: selection and evaluation on the same out-of-fold predictions (optimistic; exploration and optimizer benchmark only) |
| `n_outer_folds` | `--n-outer-folds` | 5 | Outer folds of the nested protocol |
| `n_folds` | `--n-folds` | 5 | Cross-validation folds (inner folds in the nested protocol; reduced automatically for very small classes) |
| `drop_columns` | `--drop-columns a,b` | `null` | Feature columns removed before encoding (e.g. free text). Constant columns and non-numeric identifier columns (≥ 90 % distinct values) are always removed; all removals are logged and saved in `summary.json` |
| `task_type` | `--task-type` | `auto` | `auto`, `classification`, `regression` |
| `max_models_per_algo` | `--max-models-per-algo` | 20 | 1 original model + `n − 1` feature-subset variants |
| `feature_subset_min_ratio` | `--feature-subset-min-ratio` | 0.6 | Minimum share of features in a variant |
| `feature_subset_max_ratio` | `--feature-subset-max-ratio` | 0.9 | Maximum share of features in a variant |
| `performance_metric` | `--performance-metric` | `auto` | `auto` = macro-F1 / RMSE; `accuracy`, `f1`, `mae`, `mse`, `rmse`, `r2` |
| `diversity_metric` | `--diversity-metric` | `auto` | `auto` = disagreement / ambiguity; `q_statistic`, `double_fault`, `correlation`, `rmse_difference` |
| `alpha_grid` | `--alpha-grid 0,0.5,1` | 0.0, 0.1, …, 1.0 | Fitness trade-off weights α |
| `hyperparameters_path` | `--hyperparameters` | bundled JSON | Fixed hyperparameters and stacking settings |
| `stacking_method` | `--stacking-method` | from JSON | Meta-learner; overrides the JSON value (its `params` are then ignored) |
| `stacking_inputs` | `--stacking-inputs` | `probabilities` | Classification only: stack the out-of-fold class probabilities (`predict_proba`, Ting & Witten 1999) or the predicted `labels`. Pruning and diversity always use labels |
| `fitness_normalization` | `--fitness-normalization` | `minmax` | `minmax`: P and D are rescaled to the 1st–99th percentile range of a reference sample before mixing; `none`: raw `α·P + (1 − α)·D` |
| `fitness_reference_subsets` | `--fitness-reference-subsets` | 500 | Size of the reference sample (random subsets with size uniform on {2, …, M}, seeded with `seed`) |
| `optimizer` | – | `eho` | Subset optimizer |
| `eho_variant` | `--eho-variant` | `standard` | `standard` (size-diverse init, β_EHO·centre matriarch, uniform separating, bit-flip mutation, bounded positions) or `legacy` (operators of PruneStack ≤ 2.0.0, for reproduction/benchmark only) |
| `eho_n_clans` | `--eho-n-clans` | 10 | Number of clans |
| `eho_clan_size` | `--eho-clan-size` | 10 | Elephants per clan |
| `eho_max_iterations` | `--eho-max-iterations` | 100 | Maximum iterations |
| `eho_alpha` | `--eho-alpha` | 0.5 | Matriarch influence α_EHO (start of the adaptive step α_t) |
| `eho_beta` | `--eho-beta` | 0.1 | Clan-centre factor β_EHO |
| `eho_early_stop_patience` | `--eho-early-stop-patience` | 25 | Iterations without improvement before stopping; 0 disables |
| `eho_init_inclusion_min` | `--eho-init-inclusion-min` | 0.05 | Lower bound of the per-elephant inclusion probability p_e at initialisation (`standard`) |
| `eho_init_inclusion_max` | `--eho-init-inclusion-max` | 0.5 | Upper bound of p_e (`standard`) |
| `eho_mutation_rate` | `--eho-mutation-rate` | `null` = 1/M | Bit-flip probability per bit (`standard`) |
| `eho_max_evaluations` | `--eho-max-evaluations` | `null` = unlimited | Fitness-evaluation budget (every objective call counts, including cache hits) |
| `optimization_seed` | `--optimization-seed` | = `seed` | Optimizer seed, independent of the model-pool seed |

Run-control flags (CLI only): `--output-dir`, `--skip-optimization`,
`--skip-stacking`, `--strict` (validation warnings become errors),
`--verbose`, `--quiet`.

`eho_alpha`/`eho_beta` are parameters of the optimizer and are unrelated to the
fitness weight α.

The normalisation bounds depend only on the data and `seed`, so all runs, all α
values and all optimizers of one dataset share the same fitness scale; they are
written to `optimization/fitness_normalization.json` and to
`experiment_summary.json` (`execution_metadata.fitness_normalization`). Fitness
values of PruneStack ≥ 2.1 are therefore not comparable with the raw fitness of
earlier versions (use `fitness_normalization: none` for that).

## YAML example

```yaml
seed: 1985
n_folds: 5
alpha_grid: [0.0, 0.25, 0.5, 0.75, 1.0]
performance_metric: f1
diversity_metric: q_statistic
stacking_method: hist_gradient_boosting
eho_max_iterations: 100
```

```bash
prunestack --dataset-path my_dataset --config experiment.yaml --seed 7
```

## Nested protocol (default)

```bash
prunestack --dataset-path my_dataset --seed 1985 --n-runs 30 --quiet
# -> my_dataset/results/alpha_star.json, alpha_curve.csv, heldout_performance.csv,
#    selection.csv, meta_features.json, summary.json (written last), folds/ (checkpoints)
```

Per outer fold the base models are trained once (inner OOF predictions + refit on the
outer training part); the 30 optimization runs reuse them. An interrupted run resumes
from `folds/`. Per-dataset settings for `scripts/run_all.sh` live in
`configs/datasets/<id>.yaml` (e.g. `drop_columns`). The dataset-level analysis is
`python scripts/meta_analysis.py --data-root DATA --output-dir analysis
--datasets-file configs/datasets_final.txt`. Design and rationale:
[manuscript_decisions.md](manuscript_decisions.md#5-nested-evaluation-protocol).

## Repeated optimization runs (`--protocol in_sample`)

The manuscript keeps the model pool fixed (`seed = 1985`) and repeats the
optimization 30 times with `optimization_seed = run number`. The base models
depend only on `seed` (CV folds, feature subsets, model seeds) and are therefore
identical in every run, so `--n-runs` trains them **once** and repeats only the
stochastic part (EHO, then stacking and strategy comparison on the selected subsets):

```bash
prunestack --dataset-path my_dataset --protocol in_sample --seed 1985 --n-runs 30 --quiet
# -> my_dataset/results/base_models/      shared out-of-fold predictions
# -> my_dataset/results/run_01 ... run_30 optimization_seed = 1 ... 30
```

`--optimization-seed S` shifts the seeds to `S ... S+29`. The results of run `r`
are identical to a single run with `--optimization-seed r`.

## Parallel execution

Nothing in the results depends on the degree of parallelism (every model has a
fixed `random_state`).

* **Across datasets (recommended, best throughput).** Datasets are independent;
  run one process per dataset and limit the native threads of each process:
  `scripts/run_all.sh` (local, `xargs -P`) and `scripts/slurm_array.sh` (HPC).
  `DATASETS_FILE=configs/datasets_final.txt scripts/run_all.sh DATA 8 30` restricts
  the batch to the 21 datasets of the manuscript (ids = directory names, `#` comments
  allowed; listed ids that are not found are reported).
* **Within a dataset.** `--n-jobs J` trains `J` base models in parallel
  (joblib/loky; `-1` = all cores). Useful for a single large dataset.

Do not combine many parallel processes with `--n-jobs > 1`: total cores used ≈
processes × `J`. `HistGradientBoosting*` uses OpenMP threads, so set
`OMP_NUM_THREADS=1` when running several processes side by side (the scripts do this).

## Optimizer benchmark

`scripts/benchmark_optimizers.py` compares EHO (standard and legacy) with random
search, greedy forward selection, hill climbing, simulated annealing, a genetic
algorithm and binary PSO at an equal number of fitness evaluations, on the
out-of-fold predictions written by a previous `prunestack --protocol in_sample` run:

```bash
python scripts/benchmark_optimizers.py --datasets DATA/12 DATA/52 DATA/39 \
    --alphas 0 0.5 1 --seeds 30 --budget 5000 --n-jobs 8 --out benchmark/
```

Outputs: `benchmark_runs.csv` (one row per dataset × α × optimizer × seed),
`anytime_curves.csv`, `benchmark_summary.csv` (mean/sd, Wilcoxon signed-rank with
Holm correction, paired Cohen's d_z, W/T/L against `eho_standard`),
`benchmark_overall.csv` (mean ranks and W/T/L over dataset × α blocks) and
`benchmark_config.json`. Design: [manuscript_decisions.md](manuscript_decisions.md#4-optimizer-benchmark).

## Hyperparameters JSON

All base models and the stacking meta-learner are configured in one file
(bundled copy: `src/prunestack/configs/hyperparameters.json`, documented in
[`configs/README.md`](../configs/README.md)):

```json
{
  "classification": {"LogisticRegression": {"max_iter": 1000}, "...": {}},
  "regression":     {"Ridge": {}, "...": {}},
  "stacking": {
    "classification": {"meta_learner": "nnls_mlr", "params": {}},
    "regression":     {"meta_learner": "nnls",     "params": {}}
  }
}
```

* Each task section must list exactly the six registry algorithms of that task
  (`prunestack.models.registry.list_algorithms(task_type)`); a missing or
  unknown name raises an error, so the pool cannot change silently.
* An empty object means scikit-learn defaults.
* `random_state` is injected from `seed`; do not set it in the file.
* Keys starting with `_` are comments.
