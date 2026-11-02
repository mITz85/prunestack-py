"""Ensemble size (Section 4.1) and RQ4 (Section 4.4, Table 9) of the manuscript.

Usage: python rq4_and_sizes.py [DATA_ROOT]   (layout DATA_ROOT/<id>/results/, default ../data)

Targets (per dataset, means over the 30 runs):
  alpha*_stacked = arg-max of the mean held-out stacking curve (ties -> smaller alpha)
  Delta*         = s_stacked(alpha*) - s_stacked(alpha = 1)
Six pre-specified pool descriptors are read from meta_features.json (inner predictions)
and the full stacked minus full vote gap from the reference scores.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

B = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data"
CLS = [1, 12, 49, 50, 52, 53, 87, 106]
REG = [25, 38, 39, 51, 66, 73, 74, 82, 83, 85, 90, 95, 96]
IDS = CLS + REG
DESCRIPTORS = {
    "SD of pairwise diversity": "diversity_std",
    "Full stacked minus full vote": None,
    "Mean pairwise diversity": "diversity_mean",
    "Oracle minus full ensemble": "oracle_vs_ensemble_gap",
    "SD of member performance": "model_perf_std",
    "Mean error correlation": "error_correlation_mean",
}


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for k, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[j]))
        adj[j] = running
    return adj


rows, sizes = [], []
for i in IDS:
    r = B / str(i) / "results"
    h = pd.read_csv(r / "heldout_performance.csv")
    ref = h[h.strategy.isin(["full_vote", "full_stacked"])].set_index("strategy").score
    curve = h[h.strategy == "pruned_stacked"].groupby("alpha").score.mean()
    a_star = curve.index[int(np.argmax(curve.values))]  # first maximum = smaller alpha on ties
    mf = json.loads((r / "meta_features.json").read_text())
    row = {"ds": i, "task": "cls" if i in CLS else "reg", "alpha_star": a_star,
           "delta_star": curve[a_star] - curve[1.0]}
    for name, key in DESCRIPTORS.items():
        row[name] = ref["full_stacked"] - ref["full_vote"] if key is None else mf[key]
    rows.append(row)
    sel = pd.read_csv(r / "selection.csv")
    sizes.append(sel.groupby("alpha").n_models_selected.mean().rename(i))

d = pd.DataFrame(rows)
s = pd.concat(sizes, axis=1)  # rows: alpha, columns: dataset; value = mean size over runs x folds
print("=== Ensemble size (per-dataset mean over runs and folds; M = 120)")
for a in [1.0, 0.8]:
    v = s.loc[a]
    print(f"alpha={a}: median {v.median():.1f}, mean {v.mean():.1f}, range {v.min():.1f} to {v.max():.1f}, "
          f"reduction {100 * (1 - v.median() / 120):.0f}%")

print("\n=== Task type (Mann-Whitney U, two-sided)")
c, g = d[d.task == "cls"], d[d.task == "reg"]
for t in ["delta_star", "alpha_star"]:
    print(f"{t}: median cls {c[t].median():.4f}, reg {g[t].median():.4f}, "
          f"p = {mannwhitneyu(c[t], g[t], alternative='two-sided').pvalue:.2f}")

print("\n=== Table 9: Spearman rho, pool descriptors vs targets (N = 21)")
out = []
for name in DESCRIPTORS:
    r1, p1 = spearmanr(d[name], d.delta_star)
    r2, p2 = spearmanr(d[name], d.alpha_star)
    out.append({"descriptor": name, "rho_delta": r1, "p_delta": p1, "rho_alpha": r2, "p_alpha": p2})
t9 = pd.DataFrame(out)
t9["p_delta_holm"] = holm(t9.p_delta)
t9["p_alpha_holm"] = holm(t9.p_alpha)
pd.set_option("display.width", 200)
print(t9.round(4).to_string(index=False))
