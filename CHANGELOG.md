# Changelog

All notable changes to PruneStack are documented in this file.

## [2.2.0] - 2026-10-21 (archived release 2026-11-01)

### Release packaging
- `paper/`: per-dataset results of the manuscript experiment (21 datasets, 30 runs),
  analysis scripts for all tables, statistics and figures, and their expected output.
- `.zenodo.json` and updated `CITATION.cff` for the archived release; documentation
  updated to the 120-model pool with fixed hyperparameters.

Single, clean experiment for the manuscript (RQ1–RQ4: how dataset meta-features relate to
the optimal α); rationale in `docs/manuscript_decisions.md` §5–§9.

### Added
- **Nested protocol** (`protocol: nested`, default for the manuscript; `--protocol`,
  `--n-outer-folds`): for each outer fold k the base models, fitness scaler, optimizer,
  best single model and stacking meta-learner see only the training part T_k (inner OOF
  predictions); all scores are measured on the held-out part V_k. Removes the selection
  bias of the in-sample protocol. Implementation in `experiment/nested.py`, shared helpers
  in `experiment/common.py`; fold-level checkpoints allow resumption.
- **α\* target**: arg-max of the mean held-out curve over runs (ties → smaller α), plus
  per-run α\*, the set of α within 1 SE, curve range and gain over α = 1
  (`alpha_star.json`, `alpha_curve.csv`, `heldout_performance.csv`). Used only as the
  meta-analysis target, never for performance claims.
- `scripts/meta_analysis.py`: dataset as the unit of analysis; target-free redundancy
  filter (constants, greedy |ρ| ≥ 0.9); Spearman ρ with Holm and Benjamini–Hochberg
  adjustment; leave-one-dataset-out random-forest prediction of α\* vs. median baseline;
  honest strategy comparison (α from the other datasets) with Wilcoxon + Holm and W/T/L.
- `prunestack.analysis.stats`: `paired_stats`, `holm`, `benjamini_hochberg` (shared by
  `benchmark_optimizers.py` and `meta_analysis.py`).
- **Data cleaning**: constant and identifier-like columns (nunique ≥ 0.9·n for
  non-numeric columns) are dropped automatically; `drop_columns` config/`--drop-columns`
  for user-specified columns; `summary.json` records every dropped column and the reason.
  Per-dataset YAML configs in `configs/datasets/` (e.g. `52.yaml`), picked up by
  `scripts/run_all.sh` via `DATASET_CONFIG_DIR`.

### Changed
- `FitnessScaler` reference subsets are drawn with a size uniform on {2, …, M} (was
  Bernoulli inclusion p_e ~ U(0.05, 0.5)), so small ensembles chosen by the optimizer are
  covered by the reference distribution.
- The NaN check on out-of-fold predictions only inspects rows covered by the folds.
- `scripts/run_all.sh` recognises finished nested (`summary.json`) and in-sample runs.

## [2.1.0] - 2026-10-21

Methodological changes for the manuscript; rationale in `docs/manuscript_decisions.md`.

### Breaking
- **Normalised fitness** (default `fitness_normalization: minmax`):
  `F = α·P̃ + (1 − α)·D̃`, with P and D min–max scaled to the 1st–99th percentile of
  500 random reference subsets (seed = `seed`). Fitness values are not comparable with
  2.0.0; `fitness_normalization: none` gives the raw fitness.
- **EHO defaults**: 10 clans × 10 elephants (population 100), `max_iterations` 100,
  `early_stop_patience` 25, `eho_variant: standard`.
- The fitness of an empty subset is `-inf` (was 0.0), because normalised fitness can be negative.

### Added
- `FitnessScaler`, `fit_fitness_scaler`, `sample_random_subsets` (`metrics/fitness.py`);
  `evaluate_subset` also returns `performance_score`, `performance_scaled`, `diversity_scaled`;
  `optimization/fitness_normalization.json` and `execution_metadata.fitness_normalization`.
- EHO `standard` operator set: size-diverse initialisation (p_e ~ U(0.05, 0.5)),
  matriarch = β_EHO · clan centre, uniform separating operator, bit-flip mutation (1/M),
  bounded positions; `variant="legacy"` reproduces 2.0.0 bit for bit.
- Evaluation budget: `EvaluationBudget`, `BudgetExhausted`, `eho_max_evaluations`; the
  adaptive α_t follows the used share of the budget when a budget is set.
- Config/CLI: `fitness_normalization`, `fitness_reference_subsets`, `eho_variant`,
  `eho_init_inclusion_min/max`, `eho_mutation_rate`, `eho_max_evaluations`.
- Baseline optimizers (`optimization/baselines.py`): `RandomSearch`,
  `GreedyForwardSelection`, `HillClimbing`, `SimulatedAnnealing`, `GeneticAlgorithm`, `BinaryPSO`.
- `scripts/benchmark_optimizers.py`: equal-budget benchmark (final fitness, anytime
  curves, selected-subset performance, ensemble size, Wilcoxon + Holm, Cohen's d_z, W/T/L, mean ranks).
- `configs/datasets_final.txt` (8 classification + 13 regression datasets) and
  `DATASETS_FILE` filter in `scripts/run_all.sh`.
- `optimize_alpha_grid` output gains `performance`, `performance_score`, `diversity`, `n_evaluations`.
- `--n-runs N`: repeated EHO runs (optimization seeds S..S+N-1) with a single training of
  the base models; results in `run_01 ... run_N`, shared `base_models/`.
- `--n-jobs J`: parallel training of the base-model pool (joblib); results unchanged.
- `scripts/run_all.sh` (several datasets in parallel) and `scripts/slurm_array.sh`.

### Changed
- Mean pairwise disagreement computed from per-sample vote counts in O(n·k) instead of
  O(n·k²) pairs; identical values.

## [2.0.0] - 2026-10-19

### Breaking
- Requires Python ≥ 3.15 and scikit-learn ≥ 1.9; dependency versions pinned in `requirements-lock.txt`.
- Command-line entry point is `prunestack` (`run_experiment.py` removed).
- Default stacking meta-learners: `nnls_mlr` (classification, Ting & Witten 1999) and
  `nnls` (regression, Breiman 1996). Logistic regression / ridge remain available via
  `--stacking-method` or `stacking.<task>.meta_learner` in the hyperparameters JSON.

### Removed
- Pre-computed prediction protocol (`data/precomputed.py`), per-dataset hyperparameter
  parser (`data/hpo_parser.py`), `beta_grid` / α-β grid, checkpoint/resume, 179-model padding.
- Integration tests tied to a local data directory; generated PDF/DOCX documentation.

### Changed
- Defaults match the published protocol: seed 1985, 20 models per algorithm, feature
  subsets 60–90 %, macro-F1 / RMSE, disagreement / ambiguity.
- Feature-subset variants are named `subset_01` … `subset_19`.
- Configuration parsing uses JSON/YAML only (no `eval`); unknown YAML keys raise an error.
- EHO matriarch update written in the equivalent, more readable form
  `x + β_EHO · r · (centre − x)`.
- Documentation rewritten (README, docs/, configs/README.md); `CITATION.cff` added.
- Classification stacking uses the out-of-fold class probabilities (`predict_proba`)
  as level-0 inputs by default (Ting & Witten, 1999); `stacking_inputs: labels` /
  `--stacking-inputs labels` restores label inputs. Probabilities are saved to
  `base_models/base_models_probabilities.csv`; pruning and diversity still use labels.
- Meta-features: 123 (48 dataset, 35 base-model, 40 ensemble), previously 162.
  - Performance-based meta-features (landmarkers, 1-NN error, model and ensemble
    performance) use the primary metric on the fitness scale: macro-F1 /
    RMSE skill score (previously accuracy / R²).
  - Landmarkers use the experiment folds (5-fold, stratified) and seed instead of
    a separate 3-fold split with seed 42.
  - Duplicate features removed (aliases, complements, affine copies); see
    `docs/meta_features.md`. Regression `pred_entropy_*` and
    `landmark_naive_bayes_score` no longer duplicate other features.
  - The 24 `opt_*` values are no longer meta-features (target leakage); they are
    returned by `extract_optimization_summary()` and saved to
    `optimization/optimization_summary.json`. `extract_meta_features()` no longer
    takes `optimization_results` and accepts `performance_metric`, `folds`, `seed`.

## [1.1.0] - 2026-05-20

### M7: Label Handling, Stacking HPO, Performance & Plot Fixes
- **Label Type Handling** (`loader.py`, `task_detection.py`)
  - Support string labels (e.g., "cat", "dog") with automatic LabelEncoder
  - Handle non-sequential integer labels (e.g., 1, 3, 5)
  - Regression targets: `pd.to_numeric` coercion with NaN row cleanup
  - New `detect_task_from_labels()` fallback when metrics columns are ambiguous
  - Label-based task detection passed `y` to `detect_task_type()` for fallback
- **Stacking Algorithm HPO** (`hpo_parser.py`, `stacking.py`, `experiment/runner.py`)
  - New `parse_stacking_hpo()` function reads LogisticRegression/Ridge params
  - `StackedEnsemble` accepts `meta_learner_params` dict to configure meta-learner
  - `ExperimentRunner._phase5_train_stacking()` uses HPO params from data folder
  - Filters params to valid meta-learner arguments only
- **Performance Optimization** (`diversity.py`)
  - Vectorized `pairwise_diversity_matrix()` using NumPy broadcasting
  - Classification: `(n, k, 1) != (n, 1, k)` disagreement → ~50× faster for 40 models
  - Regression: single `np.corrcoef()` call replaces O(k²) scalar correlations
  - Handles edge cases: constant predictions, single-model subsets
- **Plot Generation Fix** (`experiment/runner.py`, `visualization.py`)
  - `ExperimentRunner._phase8_save_results()` now calls `generate_experiment_report()`
  - Phase 4 saves `optimization_convergence.json` for convergence plots
  - Convergence path search expanded to check multiple locations
  - Plots directory populated: heatmap, convergence, size_vs_perf, strategy_comparison
- **Tests**: 34 new tests (396 total), all passing

## [1.0.0] - 2026-05-19

### M6: Final Polish, Documentation & Edge Cases
- **Input Validation**: Added comprehensive data quality checks (`validation.py`)
  - Dataset validation: missing values, constant features, class imbalance, shape checks
  - HPO parameter validation: algorithm checks, type verification
  - Config validation: semantic checks beyond type constraints
  - `--strict` flag for treating warnings as errors
- **Error Handling**: Custom exception hierarchy (`exceptions.py`)
  - `PruneStackError`, `DataValidationError`, `ConfigurationError`
  - `OptimizationError`, `ModelTrainingError`, `InsufficientDataError`
  - Actionable error messages with fix suggestions
- **Logging**: Professional logging system (`logging.py`)
  - Multi-level logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - Console + file output (`experiment.log`)
  - `--verbose` and `--quiet` CLI flags
- **Edge Cases**: Robust handling of boundary conditions
  - Auto-adjust CV folds for small datasets
  - High-dimensional data warnings
  - Class imbalance detection and reporting
  - Missing algorithm graceful degradation
- **Documentation**: Comprehensive documentation suite
  - Complete README with quick start, API reference, troubleshooting
  - 9 documentation files in `docs/` directory
  - API reference, configuration guide, FAQ, troubleshooting
  - Meta-features documentation (179 features)
- **Examples**: 6 usage examples in `examples/`
  - Basic usage, custom optimization, stacking-only
  - Meta-features export, batch processing, visualization
- **Performance**: Progress bars and optimizations
  - tqdm progress bars for model training and optimization
  - Efficient caching in objective function (already in M4)
- **Code Quality**: Standards compliance
  - Type hints throughout codebase
  - `__all__` exports in `__init__.py` files
  - PEP 8 compliance, consistent naming
  - CONTRIBUTING.md with guidelines
- **Release**: Production-ready packaging
  - Version bumped to 1.0.0
  - MIT License
  - Updated pyproject.toml metadata
  - Comprehensive test suite (97%+ coverage)

## [0.5.0] - M5: Stacking, Strategies & Meta-Features
- Stacking meta-learner (LogisticRegression/Ridge)
- Strategy comparison (voting vs stacking, full vs pruned)
- 179 meta-feature extraction for Layer 2 meta-learning
- ExperimentRunner with checkpointing
- HTML experiment report generation
- Results serialization and I/O

## [0.4.0] - M4: EHO Optimization
- Elephant Herding Optimization for ensemble pruning
- α-β grid optimization runner
- Fitness objective with caching
- Optimizer protocol for extensibility
- Convergence tracking and early stopping

## [0.3.0] - M3: Metrics & Fitness
- Performance metrics (accuracy, F1, R², MAE, etc.)
- Diversity metrics (disagreement, Q-statistic, double-fault)
- Multi-objective fitness function: F = α·Perf + (1-α)·Div - β·|S|/|M|
- EnsembleEvaluator for α-β grid evaluation

## [0.2.0] - M2: Base Model Training
- Cross-validated training with out-of-fold predictions
- No-leakage preprocessing pipeline (imputation + scaling)
- Per-model, per-fold detailed metrics
- Support for classification and regression tasks

## [0.1.0] - M1: Data & Configuration
- CLI interface with argparse
- YAML + CLI configuration merging
- Dataset loading (X_test.csv, y_test.csv, metrics.csv)
- HPO parameter parsing with prefix stripping
- Task type auto-detection from metrics
- Model factory with 40-model pool (4 algorithms × 10 variants)
- Feature subset generation for model diversity
