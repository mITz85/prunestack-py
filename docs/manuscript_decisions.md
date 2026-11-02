# Methodological decisions for the manuscript

This file records the decisions behind PruneStack 2.2.0 together with their reasons, so
they can be carried into the methodology section of the manuscript. Settings are given
as configuration keys (see [configuration.md](configuration.md)); all of them are written
to `summary.json` (nested), `experiment_summary.json` (in-sample) and
`benchmark_config.json` for every run.

Unchanged from 2.0.0: 120-model pool (6 algorithms × 20: original + 19 feature-subset
variants with 60–90 % of the features), fixed hyperparameters (no hyperparameter
optimization), out-of-fold predictions, macro-F1 + disagreement (classification),
RMSE skill score + normalised ambiguity (regression), α ∈ {0, 0.1, …, 1}, seed 1985,
30 repeated optimization runs (`optimization_seed` = run number).

## 1. Datasets

### 1.1 Inclusion criteria

1. n ≥ 100 rows (n = rows of `y_test.csv`, which PruneStack uses as the whole dataset);
2. classification: smallest class ≥ 10 instances (needed for stratified cross-validation
   and a stable macro-F1);
3. no time series (i.i.d. cross-validation would leak future information);
4. no target leakage and a non-trivial task (the pool must leave room for pruning).

The dataset folders, not the description spreadsheet, are authoritative (the spreadsheet
has no path for several existing folders). Criteria 1–3 are checked on the data;
criterion 4 was screened with the results of an earlier experiment with 4 models
(macro-F1 / explained variance of every model).

### 1.2 Final list: 8 classification + 13 regression = 21 datasets

The strictly clean set is used: every dataset that is doubtful under criterion 4 is
excluded instead of repaired. List: [`configs/datasets_final.txt`](../configs/datasets_final.txt).

| Task | ID (n; smallest class) |
|---|---|
| Classification | 1 (1364; 49), 12 (1004; 52), 49 (229; 21), 50 (324; 17), 52 (211; 32), 53 (292; 103), 87 (3175; 157), 106 (126; 10) |
| Regression | 25 (392), 38 (483), 39 (610), 51 (209), 66 (961), 73 (8012), 74 (15866), 82 (172), 83 (256), 85 (3028), 90 (363), 95 (1203), 96 (4402) |

Notes for the manuscript:

* 73 and 74 share the 30 features and the target (hotel booking ADR) but have no common
  row; they are most likely the two hotels (city / resort) of one source. Both are kept
  and the shared domain is mentioned.
* 106 is at the class-size limit (smallest class exactly 10).

### 1.3 Excluded datasets

| Reason | IDs |
|---|---|
| n < 100 (classification) | 2, 6, 16, 20, 22, 28, 34, 40, 41, 44, 45, 46, 55, 63, 64, 68, 69, 77, 80, 84, 97, 98, 100, 102, 104, 108, 109, 110 |
| n < 100 (regression) | 19, 24, 31, 33, 37, 54, 56, 57, 58, 60, 62, 65, 67, 70, 71, 72, 75, 79, 86, 88, 91, 111 |
| Smallest class < 10 | 13 (min 2), 61 (min 4), 99 (min 1, 11 classes), 101 (min 1) |
| Time series | 3, 4, 32 (no data folder) |
| Suspected target leakage | 8, 9, 10, 11 (Busan housing, same source: tree MAE ≈ 2–12 at sd(y) ≈ 15 000, Ridge explained variance 0.999); 78 (Ridge explained variance 0.999) |
| Trivial task | 47 (macro-F1 = 1.00 for all models), 48 (1.00 / 1.00 / 1.00 / 0.95) |
| No signal | 81 (explained variance ≈ 0 for all models; extreme target outliers) |
| Duplicates | 22 = 69; 70/71/72 (same rows, different targets) — also n < 100 |

Leakage and triviality are hypotheses from model behaviour; the leaking features were not
identified. Excluding these datasets keeps the meta-dataset free of tasks on which every
subset performs (almost) identically.

## 2. Normalised fitness

**Problem.** P(S) and D(S) both lie in [0, 1] but occupy different ranges (e.g. on dataset
12: macro-F1 ≈ 0.76, disagreement ≈ 0.27). In `α·P + (1 − α)·D` the larger and more
variable term dominates, the fitness grows almost linearly with α and the best α was
almost always 1.0, so α was not a meaningful trade-off and `opt_best_alpha` a poor
meta-target.

**Decision** (`fitness_normalization: minmax`):

```
F(S) = α · P̃(S) + (1 − α) · D̃(S)
P̃(S) = (P(S) − P_lo) / (P_hi − P_lo),   D̃(S) = (D(S) − D_lo) / (D_hi − D_lo)
```

* `[lo, hi]` = 1st and 99th percentile of P and D over a reference sample of 500 random
  subsets (`fitness_reference_subsets`). Since 2.2.0 the size k of a reference subset is
  uniform on {2, …, M} and its k models are drawn without replacement. (2.1.0 drew a
  Bernoulli inclusion probability per subset; the resulting size distribution
  under-represented the small subsets the optimizer actually selects, so P̃ and D̃ of
  selected subsets were often far above 1.) Optimized subsets can still exceed the
  reference range, because they are better than random subsets of the same size.
* The reference sample depends only on the data and `seed` (1985), never on the optimizer
  seed: all α values, all 30 runs and all optimizers of a dataset share one fitness scale.
* The map is affine and not clipped. For a fixed α it does not change the ranking of
  subsets, and an optimizer can exceed the reference range (P̃, D̃ > 1).
* Percentiles instead of min/max make the bounds robust to single extreme subsets.
* The empty subset has fitness −∞ (normalised fitness can be negative).

Bounds are saved in `optimization/fitness_normalization.json`. Fitness values of 2.1.0
are not comparable with 2.0.0; `fitness_normalization: none` reproduces the raw fitness.

## 3. Elephant Herding Optimization

### 3.1 Diagnosis of the 2.0.0 implementation

On datasets 8 and 12 the whole gain appeared in the first 10–13 iterations, early stopping
ended the search after 19–36 of 50 iterations, and the selected ensembles had 47–68 of 120
models for every α. Causes in the code:

1. Standard-normal initialisation with threshold x ≥ 0 makes every bit Bernoulli(0.5): all
   initial subsets have ≈ 60 ± 5.5 models; small ensembles were never explored.
2. Clan updating with α_t → 1 is a pure contraction towards the matriarch; positions
   collapse and bits stop changing.
3. The matriarch moved towards the clan centre (β·(centre − x)·r), unlike the original
   operator (matriarch = β · centre).
4. The separating operator re-drew the worst elephant from N(0, 1), i.e. again a random
   ≈ 60-model subset that is rarely competitive.
5. No mutation, hence no escape after the collapse.
6. Small budget (≈ 1 000–1 850 evaluations) and patience 10.

Part of the weak convergence belongs to the problem: for majority vote or averaging over
≈ 60 models, swapping a few models hardly changes the ensemble output, so the fitness
landscape is flat around large ensembles.

### 3.2 Decision: `eho_variant: standard`

Binary EHO (Wang, Deb & Coelho, 2015). An elephant has a continuous position
x ∈ [−B, B]^M (B = 4); model j is selected when sigmoid(x_j) ≥ 0.5 (x_j ≥ 0). An empty
subset is repaired to one random model.

| Operator | Definition | Reason |
|---|---|---|
| Initialisation | p_e ~ U(0.05, 0.5) per elephant, x_ej ~ N(Φ⁻¹(p_e), 1) ⇒ P(bit = 1) = p_e | Covers small and medium ensembles |
| Clan updating (Eq. 1) | x ← x + α_t · (x_matriarch − x) · r, r ~ U(0, 1)^M | Original operator |
| Adaptive step | α_t = α_EHO + (1 − α_EHO) · t/T (T = max_iterations, or the used share of the evaluation budget) | Exploration early, exploitation late |
| Matriarch (Eq. 2) | x_matriarch ← β_EHO · x_centre | Faithful to the cited method |
| Mutation | Each bit of a non-matriarch elephant flipped (x_j ← −x_j) with probability 1/M | Escape after contraction |
| Separating (Eq. 4) | Worst elephant of each clan re-sampled uniformly in [−B, B]^M | Original operator |
| Bounds | Positions clipped to [−B, B] | Keeps bits flippable |

Settings: 10 clans × 10 elephants (population 100), α_EHO = 0.5, β_EHO = 0.1,
`max_iterations` 100, `early_stop_patience` 25. The best subset ever evaluated is
returned. α_EHO and β_EHO are optimizer parameters, unrelated to the fitness weight α.
`eho_variant: legacy` reproduces 2.0.0 exactly and is used only in the benchmark.

## 4. Optimizer benchmark

**Purpose.** Show whether EHO is a justified choice or whether simpler search reaches the
same fitness — the obvious reviewer question. Diagnostic study, separate from the main
experiment ([`scripts/benchmark_optimizers.py`](../scripts/benchmark_optimizers.py)).

**Design.**

* Same fitness for all optimizers (normalised, scaler of the main experiment with seed 1985),
  same 120-model out-of-fold predictions, same number of fitness evaluations (every
  objective call counts, cache hits included). The budget is the only stopping criterion
  (EHO early stopping off); greedy forward selection is deterministic and stops at its
  first step without improvement.
* Optimizers: EHO standard (reference), EHO legacy, random search (lower bound), greedy
  forward selection (ordered pruning), first-improvement hill climbing, simulated annealing
  (single-bit flips, geometric cooling 0.01 → 1e-5), generational GA (population 100,
  tournament 3, uniform crossover 0.9, bit-flip 1/M, one elite), binary PSO (swarm 100,
  w = 0.7298, c1 = c2 = 1.49618, v_max = 4). All stochastic methods use the size-diverse
  initialisation of §3.2. Parameters are literature defaults, not tuned.
* Suggested run: a subset of datasets of both tasks, α ∈ {0, 0.5, 1}, 30 seeds,
  budget 5 000 evaluations.

**Measures.** Final best fitness; anytime curve (best-so-far fitness at log-spaced
evaluation counts); macro-F1 / RMSE of the selected subset (better OOF fitness need not
mean better generalisation); ensemble size; runtime.

**Statistics.** Per dataset × α: mean/sd, Wilcoxon signed-rank test against EHO standard
over paired seeds with Holm correction, paired Cohen's d_z, win/tie/loss. Over all
dataset × α blocks: mean ranks and Wilcoxon test on block means.

**Interpretation.** EHO standard ≈ GA/PSO > random search ⇒ EHO is a legitimate choice.
Random search or greedy ≈ EHO ⇒ the optimizer adds no value; this is then reported
honestly and the fitness design is reconsidered.

Status: implemented and unit-tested; not yet run on the 21 datasets, so no results are
claimed here.

## 5. Nested evaluation protocol

**Problem (manuscript ≤ 2.1).** Subsets were selected on the out-of-fold (OOF)
predictions of the base models and the selected ensemble was scored on the *same* OOF
predictions. Selection then fits the noise of those rows (selection bias), so "pruned
beats full" and "ensemble beats best single model" were optimistic, and the text
incorrectly called this a hold-out protocol. The best single model was also chosen and
scored on the same predictions.

**Decision** (`protocol: nested`, default; `experiment/nested.py`):

1. Outer stratified (classification) / plain (regression) 5-fold split, seed 1985.
2. For outer fold k: inner 5-fold OOF predictions of all 120 models on the training part
   T_k; every model refitted on T_k predicts the held-out part V_k.
3. Fitness normalization bounds, EHO selection (30 runs × 11 α), the stacking
   meta-learner and the choice of the best single model use only T_k.
4. The selected subset is evaluated once on V_k. Held-out predictions of the 5 folds are
   concatenated, so every strategy gets one cross-validated prediction per row and the
   metric (macro-F1, RMSE) is computed once on all n rows.

Base-model hyperparameters are fixed (no tuning anywhere), so no further nesting level
is needed. The cost is ≈ 6× base-model training and 5× optimization per dataset
compared with the in-sample protocol. `protocol: in_sample` remains for exploration and
for the optimizer benchmark (§4), where only the search quality on fixed predictions
matters.

## 6. Meta-target α*

* α and β = 1 − α are one parameter; the manuscript reports only α.
* **Old target:** α maximizing the *fitness*. Fitness values at different α are different
  functions and are not comparable, so arg-max over α measured the scale of P versus D,
  not a trade-off (α → 1 artefact, §2).
* **New target:** for each run, the held-out score (macro-F1 / RMSE skill) of the pruned
  majority-vote/averaging ensemble selected with weight α gives a curve over the 11 α
  values; α* = arg-max of the mean curve over the 30 runs (ties → smaller α). Also stored:
  α* per run, the α values within one standard error of the maximum, the curve range and
  the gain over α = 1 (`alpha_star.json`, `alpha_curve.csv`). The same target is computed
  for pruned stacking.
* α* uses the held-out rows of the dataset, therefore it is only a *target* of the
  meta-analysis. Performance claims never use a dataset's own α* (§7).

## 7. Dataset-level meta-analysis

`scripts/meta_analysis.py`; the statistical unit is the dataset (run means). Analyses are
reported for classification (8), regression (13) and all 21 datasets.

* **Strategies (RQ1, RQ2, RQ4).** The α of a dataset is the median α* of the *other*
  datasets of the same task (leave-one-dataset-out), so the comparison is honest.
  Pruned vote vs full vote, pruned vote vs best single, pruned vote vs α = 1, pruned
  stacking vs full stacking / best single / pruned vote: Wilcoxon signed-rank test over
  datasets, Holm correction over the comparisons, Cohen's d_z, W/T/L (tie: |Δ| ≤ 0.001 on
  the [0, 1] score scale).
* **Associations (RQ3).** Spearman ρ between each meta-feature and α*; Holm (FWER) and
  Benjamini–Hochberg (FDR) adjusted over the meta-features. With 8–21 datasets the power
  is low: the analysis is reported as exploratory and the adjusted p-values are primary.
* **Prediction (RQ3).** Leave-one-dataset-out random forest (500 trees, min leaf 2,
  max_features √p, not tuned) predicting α* versus the median baseline; loss = regret,
  i.e. held-out score lost relative to the dataset's α*, compared with the Wilcoxon test.
* **Redundant meta-features.** Target-free filter applied inside every training set:
  constant features are dropped, then, in the fixed feature order, every feature with
  |Spearman ρ| ≥ 0.9 to an already kept feature (`meta_feature_filter.csv` lists the
  reason for each removal). This complements the removal of exact/affine copies in
  `docs/meta_features.md`.
* Meta-features are computed from X, y and the cross-validated held-out predictions of
  the pool; the optimization summary is never used (target leakage).

## 8. Data cleaning before modelling

The loader removes constant columns and non-numeric identifier columns (≥ 90 % distinct
values) before label encoding, and columns listed in `drop_columns`
(`configs/datasets/<id>.yaml`). In the final 21 datasets this removes: 1 `Net Income
Flag` (constant); 52 `name` (identifier), `national` (constant) and `bio` (free text:
e-mails, phone numbers; listed manually); 95 two constant ratio columns; 96
`Previous_pdn` (constant). Removals are saved in `summary.json`. `ReservationStatusDate`
(73, 74) is kept: an ISO date whose label encoding is chronological, i.e. an ordinal
time feature, not an identifier; ADR is fixed at booking, so the reservation status is
not a leak of the target.

## 9. Running the experiment

```bash
DATASETS_FILE=configs/datasets_final.txt scripts/run_all.sh DATA 8 30
python scripts/meta_analysis.py --data-root DATA --output-dir analysis \
    --datasets-file configs/datasets_final.txt
# optimizer benchmark (diagnostic, in-sample predictions):
DATASETS_FILE=configs/datasets_final.txt scripts/run_all.sh DATA 8 1 --protocol in_sample
python scripts/benchmark_optimizers.py --datasets DATA/12 DATA/52 DATA/39 \
    --alphas 0 0.5 1 --seeds 30 --budget 5000 --n-jobs 8 --out benchmark/
```

## References

* Wang, G.-G., Deb, S., & Coelho, L. d. S. (2015). Elephant herding optimization. *ISCBI 2015*, 1–5.
* Kennedy, J., & Eberhart, R. C. (1997). A discrete binary version of the particle swarm algorithm. *IEEE SMC*, 4104–4108.
* Clerc, M., & Kennedy, J. (2002). The particle swarm — explosion, stability, and convergence. *IEEE TEC*, 6(1), 58–73.
* Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Optimization by simulated annealing. *Science*, 220(4598), 671–680.
* Krogh, A., & Vedelsby, J. (1995). Neural network ensembles, cross validation, and active learning. *NIPS 7*, 231–238.
