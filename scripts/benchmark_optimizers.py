#!/usr/bin/env python
"""Equal-budget benchmark of subset-selection optimizers (diagnostic study).

Every optimizer receives the same fitness function (normalised α·P̃ + (1 − α)·D̃,
with the reference-sample scaler of the main experiment, seed 1985) and the same
number of fitness evaluations. The only stopping criterion is the budget
(EHO early stopping is disabled), except greedy forward selection, which is
deterministic and stops at its first step without improvement.

Inputs (per dataset)
    DATASET_DIR                 X_test.csv, y_test.csv, metrics.csv (as for prunestack)
    PREDICTIONS_CSV (optional)  out-of-fold predictions of the model pool;
                                default DATASET_DIR/results/base_models/base_models_predictions.csv

Outputs (in --out)
    benchmark_runs.csv     one row per dataset × α × optimizer × seed
    anytime_curves.csv     best-so-far fitness at log-spaced evaluation checkpoints
    benchmark_summary.csv  per dataset × α × optimizer: mean/sd and paired tests vs. the reference
    benchmark_overall.csv  across dataset × α blocks: mean rank, W/T/L, Wilcoxon on block means
    benchmark_config.json  all settings, for the manuscript

Example
    python scripts/benchmark_optimizers.py --datasets DATA/12 DATA/49 --seeds 30 \
        --budget 5000 --alphas 0 0.5 1 --n-jobs 8 --out benchmark/
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata

from prunestack import __version__
from prunestack.analysis.stats import holm, paired_stats
from prunestack.data.loader import load_dataset
from prunestack.metrics.fitness import evaluate_subset, fit_fitness_scaler
from prunestack.optimization import (
    BinaryPSO,
    EHOOptimizer,
    EvaluationBudget,
    FitnessObjective,
    GeneticAlgorithm,
    GreedyForwardSelection,
    HillClimbing,
    RandomSearch,
    SimulatedAnnealing,
)

REFERENCE = "eho_standard"
UNLIMITED = 10**9  # iteration cap that never binds; the evaluation budget stops the search


def build_optimizers(budget: int) -> dict[str, object]:
    """All optimizers with default settings and an identical evaluation budget."""
    common = {"max_evaluations": budget, "max_iterations": UNLIMITED}
    eho = {**common, "early_stop_patience": 0}
    return {
        "eho_standard": EHOOptimizer(variant="standard", **eho),
        "eho_legacy": EHOOptimizer(variant="legacy", **eho),
        "random_search": RandomSearch(**common),
        "greedy_forward": GreedyForwardSelection(**common),
        "hill_climbing": HillClimbing(**common),
        "simulated_annealing": SimulatedAnnealing(**common),
        "genetic_algorithm": GeneticAlgorithm(**common),
        "binary_pso": BinaryPSO(**common),
    }


def checkpoints(budget: int, n_points: int = 40) -> np.ndarray:
    return np.unique(np.round(np.geomspace(1, budget, n_points)).astype(int))


def run_cell(name, y, preds, task_type, scaler, alpha, seed, budget, metric, diversity_metric):
    """All optimizers for one dataset × α × seed; a fresh objective (empty cache) per optimizer."""
    runs, curves = [], []
    marks = checkpoints(budget)
    for opt_name, optimizer in build_optimizers(budget).items():
        objective = FitnessObjective(y, preds, alpha=alpha, task_type=task_type, performance_metric=metric,
                                     diversity_metric=diversity_metric, scaler=scaler)
        tracker = EvaluationBudget(objective, record_trace=True)
        start = time.perf_counter()
        result = optimizer.optimize(tracker, preds.shape[1], seed=seed)
        elapsed = time.perf_counter() - start
        detail = evaluate_subset(y, preds, result.best_indices, alpha, task_type, performance_metric=metric,
                                 diversity_metric=diversity_metric, scaler=scaler, return_predictions=False)
        runs.append({
            "dataset": name, "optimizer": opt_name, "alpha": alpha, "seed": seed,
            "best_fitness": result.best_fitness, "n_selected": result.n_selected,
            "performance": detail["performance"], "performance_score": detail["performance_score"],
            "diversity": detail["diversity"], "n_evaluations": tracker.n_evaluations,
            "n_iterations": result.n_iterations, "time_s": elapsed,
        })
        trace = np.asarray(tracker.trace)
        for m in marks:
            curves.append({"dataset": name, "optimizer": opt_name, "alpha": alpha, "seed": seed,
                           "evaluations": int(m), "best_fitness": float(trace[min(m, len(trace)) - 1])})
    return runs, curves


def summarise(runs: pd.DataFrame, tol: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (dataset, alpha), block in runs.groupby(["dataset", "alpha"], sort=False):
        wide = block.pivot(index="seed", columns="optimizer", values="best_fitness")
        stats = []
        for opt_name, grp in block.groupby("optimizer", sort=False):
            row = {"dataset": dataset, "alpha": alpha, "optimizer": opt_name, "n_seeds": len(grp)}
            for col in ("best_fitness", "n_selected", "performance", "diversity", "n_evaluations", "time_s"):
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_sd"] = grp[col].std(ddof=1)
            if opt_name != REFERENCE:
                row.update(paired_stats(wide[REFERENCE].to_numpy(), wide[opt_name].to_numpy(), tol))
            stats.append(row)
        block_df = pd.DataFrame(stats)
        tested = block_df["optimizer"] != REFERENCE
        block_df.loc[tested, "wilcoxon_p_holm"] = holm(block_df.loc[tested, "wilcoxon_p"]).to_numpy()
        rows.append(block_df)
    summary = pd.concat(rows, ignore_index=True)

    # Across dataset × α blocks (Demšar, 2006): ranks and W/T/L of the block means.
    means = summary.pivot_table(index=["dataset", "alpha"], columns="optimizer", values="best_fitness_mean")
    ranks = means.apply(lambda r: pd.Series(rankdata(-r.to_numpy()), index=r.index), axis=1)
    overall = []
    for opt_name in means.columns:
        row = {"optimizer": opt_name, "n_blocks": len(means), "mean_rank": ranks[opt_name].mean()}
        if opt_name != REFERENCE:
            row.update(paired_stats(means[REFERENCE].to_numpy(), means[opt_name].to_numpy(), tol))
        overall.append(row)
    return summary, pd.DataFrame(overall).sort_values("mean_rank")


def parse_dataset(spec: str, results_subdir: str) -> tuple[str, Path, Path]:
    dataset_dir, _, predictions = spec.partition(":")
    dataset_dir = Path(dataset_dir)
    predictions = Path(predictions) if predictions else dataset_dir / results_subdir / "base_models" / "base_models_predictions.csv"
    return dataset_dir.name, dataset_dir, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", required=True, help="DATASET_DIR[:PREDICTIONS_CSV] ...")
    parser.add_argument("--results-subdir", default="results")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    parser.add_argument("--seeds", type=int, default=30, help="optimizer seeds 1..N")
    parser.add_argument("--budget", type=int, default=5000, help="fitness evaluations per run")
    parser.add_argument("--seed", type=int, default=1985, help="data/scaler seed (as in the main experiment)")
    parser.add_argument("--fitness-normalization", default="minmax", choices=["minmax", "none"])
    parser.add_argument("--fitness-reference-subsets", type=int, default=500)
    parser.add_argument("--performance-metric", default=None)
    parser.add_argument("--diversity-metric", default="auto")
    parser.add_argument("--tie-tolerance", type=float, default=1e-9)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    all_runs, all_curves, scalers = [], [], {}
    for spec in args.datasets:
        name, dataset_dir, predictions_csv = parse_dataset(spec, args.results_subdir)
        bundle = load_dataset(dataset_dir, seed=args.seed)
        y = np.asarray(bundle.y)
        preds = pd.read_csv(predictions_csv).to_numpy(dtype=float)
        if len(preds) != len(y):
            raise ValueError(f"{name}: {len(preds)} prediction rows but {len(y)} targets")
        scaler = fit_fitness_scaler(y, preds, bundle.task_type, performance_metric=args.performance_metric,
                                    diversity_metric=args.diversity_metric, method=args.fitness_normalization,
                                    n_subsets=args.fitness_reference_subsets, seed=args.seed)
        scalers[name] = {"task_type": bundle.task_type, "n_samples": len(y), "n_models": preds.shape[1],
                         **scaler.to_dict()}
        print(f"[{name}] {bundle.task_type}, n={len(y)}, M={preds.shape[1]}; "
              f"{len(args.alphas)} alphas x {args.seeds} seeds x 8 optimizers, budget {args.budget}", flush=True)
        start = time.perf_counter()
        cells = Parallel(n_jobs=args.n_jobs)(
            delayed(run_cell)(name, y, preds, bundle.task_type, scaler, alpha, seed, args.budget,
                              args.performance_metric, args.diversity_metric)
            for alpha in args.alphas for seed in range(1, args.seeds + 1)
        )
        for runs, curves in cells:
            all_runs.extend(runs)
            all_curves.extend(curves)
        print(f"[{name}] done in {time.perf_counter() - start:.0f} s", flush=True)

    runs_df = pd.DataFrame(all_runs)
    runs_df.to_csv(args.out / "benchmark_runs.csv", index=False)
    pd.DataFrame(all_curves).to_csv(args.out / "anytime_curves.csv", index=False)
    summary, overall = summarise(runs_df, args.tie_tolerance)
    summary.to_csv(args.out / "benchmark_summary.csv", index=False)
    overall.to_csv(args.out / "benchmark_overall.csv", index=False)
    config = {
        "prunestack_version": __version__, "python": platform.python_version(),
        "reference_optimizer": REFERENCE, "budget": args.budget, "seeds": list(range(1, args.seeds + 1)),
        "alphas": args.alphas, "data_seed": args.seed, "tie_tolerance": args.tie_tolerance,
        "optimizers": {k: repr(v) for k, v in build_optimizers(args.budget).items()},
        "fitness_scalers": scalers,
    }
    (args.out / "benchmark_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
