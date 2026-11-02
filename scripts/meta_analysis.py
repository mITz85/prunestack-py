#!/usr/bin/env python
"""Dataset-level meta-analysis of the nested experiment (manuscript RQ1-RQ4).

Reads ``<data-root>/<id>/results/{summary,alpha_star,meta_features}.json`` and
``alpha_curve.csv`` written by ``prunestack --protocol nested`` and answers:

RQ1/RQ2  Do pruned ensembles (majority vote / stacking) beat the full pool and the
         best single model on held-out rows?  The alpha of a dataset is chosen
         *without* its own held-out rows: the median alpha* of the other datasets of
         the same task (leave-one-dataset-out), so the comparison is not optimistic.
RQ3      Which meta-features are associated with alpha*?  Spearman correlations,
         Holm (FWER) and Benjamini-Hochberg (FDR) adjusted over the meta-features.
RQ4      Can alpha* be predicted from meta-features?  Leave-one-dataset-out random
         forest versus the median baseline; the loss is the *regret*, i.e. the
         held-out score lost relative to the dataset's oracle alpha*.

The statistical unit is the dataset (scores are means over the optimization runs),
never the run, fold or alpha. Analyses are reported per task and for both tasks
together (all scores are on the [0, 1] scale: macro-F1, RMSE skill 1 - RMSE/sd(y)).

Redundant meta-features are removed without looking at the target, separately
inside every leave-one-out training set: features constant over the datasets and,
greedily in the fixed feature order, features with |Spearman rho| >= --redundancy
to an already kept feature.

Usage::

    python scripts/meta_analysis.py --data-root DATA --output-dir analysis \
        [--datasets-file configs/datasets_final.txt]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor

from prunestack.analysis.stats import benjamini_hochberg, holm, paired_stats

logger = logging.getLogger("meta_analysis")
TARGETS = ("pruned_vote", "pruned_stacked")
GROUPS = ("classification", "regression", "all")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def load_results(data_root: Path, ids: list[str] | None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """Per-dataset targets, meta-features and held-out alpha curves of completed datasets."""
    targets, features, curves = [], {}, {}
    for summary_path in sorted(data_root.glob("*/results/summary.json")):
        results = summary_path.parent
        dataset = results.parent.name
        if ids is not None and dataset not in ids:
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("protocol") != "nested":
            continue
        alpha = json.loads((results / "alpha_star.json").read_text())
        row = {"dataset": dataset, "task_type": alpha["task_type"], "n_samples": summary["n_samples"],
               "n_features": summary["n_features"], "n_runs": alpha["n_runs"]}
        row |= {f"{name}_score": score for name, score in alpha["reference_scores"].items()}
        for target in TARGETS:
            t = alpha[target]
            row |= {f"{target}_alpha_star": t["alpha_star"], f"{target}_score_at_alpha_star": t["score_at_alpha_star"],
                    f"{target}_curve_range": t["curve_range"], f"{target}_gain_over_alpha_1": t.get("gain_over_alpha_1")}
        targets.append(row)
        features[dataset] = json.loads((results / "meta_features.json").read_text())
        curves[dataset] = pd.read_csv(results / "alpha_curve.csv").set_index("alpha")
    if ids is not None:
        missing = sorted(set(ids) - {r["dataset"] for r in targets})
        if missing:
            logger.warning("No completed nested results for: %s", missing)
    if not targets:
        raise SystemExit(f"no completed nested results below {data_root}")
    target_df = pd.DataFrame(targets).set_index("dataset")
    feature_df = pd.DataFrame.from_dict(features, orient="index").loc[target_df.index].astype(float)
    return target_df, feature_df, curves


# ----------------------------------------------------------------------------
# Meta-features
# ----------------------------------------------------------------------------

def redundancy_filter(X: pd.DataFrame, threshold: float) -> tuple[list[str], dict[str, str]]:
    """Kept columns and the reason every other column was removed (target-free)."""
    removed: dict[str, str] = {}
    candidates = []
    for col in X.columns:
        if X[col].nunique(dropna=True) <= 1:
            removed[col] = "constant"
        else:
            candidates.append(col)
    rho = X[candidates].rank().corr().abs() if candidates else pd.DataFrame()
    kept: list[str] = []
    for col in candidates:
        partner = next((k for k in kept if rho.loc[col, k] >= threshold), None)
        if partner is None:
            kept.append(col)
        else:
            removed[col] = f"|rho|={rho.loc[col, partner]:.2f} with {partner}"
    return kept, removed


def correlations(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        res = spearmanr(X[col], y)
        rows.append({"meta_feature": col, "spearman_rho": float(res.statistic), "p": float(res.pvalue)})
    out = pd.DataFrame(rows).dropna(subset=["p"])
    out["p_holm"] = holm(out["p"]).to_numpy()
    out["p_bh"] = benjamini_hochberg(out["p"]).to_numpy()
    return out.sort_values("p").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Leave-one-dataset-out alpha
# ----------------------------------------------------------------------------

def snap(value: float, grid: np.ndarray) -> float:
    return float(grid[np.argmin(np.abs(grid - value))])


def loo_alphas(X: pd.DataFrame, y: pd.Series, grid: np.ndarray, threshold: float, seed: int) -> pd.DataFrame:
    """Median-baseline and random-forest alpha for every dataset, fitted without it."""
    rows = []
    for dataset in X.index:
        train = X.index != dataset
        kept, _ = redundancy_filter(X.loc[train], threshold)
        model = RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=seed)
        model.fit(X.loc[train, kept], y[train])
        rows.append({"dataset": dataset,
                     "alpha_median": snap(float(np.median(y[train])), grid),
                     "alpha_meta_model": snap(float(model.predict(X.loc[[dataset], kept])[0]), grid),
                     "n_meta_features_used": len(kept)})
    return pd.DataFrame(rows).set_index("dataset")


def score_at(curves: dict[str, pd.DataFrame], dataset: str, target: str, alpha: float) -> float:
    curve = curves[dataset][f"{target}_mean"]
    return float(curve.iloc[np.argmin(np.abs(curve.index.to_numpy() - alpha))])


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def analyse_group(group: str, targets: pd.DataFrame, features: pd.DataFrame, curves: dict[str, pd.DataFrame],
                  args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    if group != "all":
        targets, features = targets[targets.task_type == group], features.loc[targets.task_type == group]
    else:
        features = features.assign(task_is_classification=(targets.task_type == "classification").astype(float))
    n = len(targets)
    out: dict[str, list[pd.DataFrame]] = {k: [] for k in ("filter", "correlations", "loo", "scores", "comparison")}
    if n < 4:
        logger.warning("%s: only %d dataset(s), skipped", group, n)
        return {}
    kept, removed = redundancy_filter(features, args.redundancy)
    out["filter"].append(pd.DataFrame(
        [{"group": group, "meta_feature": c, "kept": c in kept, "reason": removed.get(c, "")} for c in features.columns]))

    scores = pd.DataFrame(index=targets.index)
    for ref in ("full_vote", "full_stacked", "best_single"):
        scores[ref] = targets[f"{ref}_score"]
    for target in TARGETS:
        y = targets[f"{target}_alpha_star"]
        grid = curves[targets.index[0]].index.to_numpy(dtype=float)
        out["correlations"].append(correlations(features[kept], y).assign(group=group, target=target))

        loo = loo_alphas(features, y, grid, args.redundancy, args.seed)
        if group == "all":  # median of the other datasets of the same task
            for d in targets.index:
                same = (targets.task_type == targets.task_type[d]) & (targets.index != d)
                loo.loc[d, "alpha_median"] = snap(float(np.median(y[same])), grid)
        loo["alpha_star"] = y
        for alpha_col in ("alpha_star", "alpha_median", "alpha_meta_model"):
            loo[f"score_{alpha_col}"] = [score_at(curves, d, target, loo.loc[d, alpha_col]) for d in loo.index]
        loo["score_alpha_1"] = [score_at(curves, d, target, 1.0) for d in loo.index]
        for alpha_col in ("alpha_median", "alpha_meta_model"):
            loo[f"regret_{alpha_col}"] = loo["score_alpha_star"] - loo[f"score_{alpha_col}"]
            loo[f"abs_error_{alpha_col}"] = (loo[alpha_col] - loo["alpha_star"]).abs()
        out["loo"].append(loo.reset_index().assign(group=group, target=target))
        scores[f"{target}_loo_median"] = loo["score_alpha_median"]
        scores[f"{target}_loo_meta_model"] = loo["score_alpha_meta_model"]
        scores[f"{target}_alpha_1"] = loo["score_alpha_1"]

    comparisons = [  # (reference, competitor): positive difference = reference better
        ("pruned_vote_loo_median", "full_vote"),
        ("pruned_vote_loo_median", "best_single"),
        ("pruned_vote_loo_median", "pruned_vote_alpha_1"),
        ("pruned_stacked_loo_median", "full_stacked"),
        ("pruned_stacked_loo_median", "best_single"),
        ("pruned_stacked_loo_median", "pruned_vote_loo_median"),
        ("pruned_vote_loo_meta_model", "pruned_vote_loo_median"),
        ("pruned_stacked_loo_meta_model", "pruned_stacked_loo_median"),
    ]
    comp = pd.DataFrame([{"group": group, "reference": ref, "competitor": other,
                          **paired_stats(scores[ref].to_numpy(), scores[other].to_numpy(), args.tie_tolerance)}
                         for ref, other in comparisons])
    comp["wilcoxon_p_holm"] = holm(comp["wilcoxon_p"]).to_numpy()
    out["comparison"].append(comp)
    out["scores"].append(scores.reset_index().assign(group=group, task_type=targets.task_type.to_numpy()))
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items() if v}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets-file", type=Path, help="restrict to the dataset ids listed (one per line, '#' comments)")
    parser.add_argument("--redundancy", type=float, default=0.9, help="|Spearman rho| above which a meta-feature is redundant")
    parser.add_argument("--tie-tolerance", type=float, default=1e-3, help="|score difference| counted as a tie in W/T/L")
    parser.add_argument("--seed", type=int, default=1985)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    ids = None
    if args.datasets_file:
        ids = [line.split("#")[0].strip() for line in args.datasets_file.read_text().splitlines()]
        ids = [i for i in ids if i]
    targets, features, curves = load_results(args.data_root, ids)
    logger.info("%d datasets: %s", len(targets), targets.task_type.value_counts().to_dict())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.output_dir / "dataset_targets.csv")
    results: dict[str, list[pd.DataFrame]] = {}
    for group in GROUPS:
        for name, df in analyse_group(group, targets, features, curves, args).items():
            results.setdefault(name, []).append(df)
    files = {"filter": "meta_feature_filter.csv", "correlations": "meta_feature_correlations.csv",
             "loo": "loo_alpha_predictions.csv", "scores": "strategy_scores.csv",
             "comparison": "strategy_comparison.csv"}
    for name, frames in results.items():
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir / files[name], index=False)
    if "comparison" in results:
        comp = pd.concat(results["comparison"], ignore_index=True)
        cols = ["group", "reference", "competitor", "n", "mean_diff_vs_ref", "cohens_dz",
                "wilcoxon_p", "wilcoxon_p_holm", "ref_wins", "ties", "ref_losses"]
        logger.info("Strategy comparison (dataset = unit):\n%s", comp[cols].to_string(index=False))
    logger.info("Results written to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
