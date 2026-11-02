"""Paired statistics shared by the analysis scripts.

The statistical unit is always the independent replicate of the comparison: the
dataset for the manuscript (``scripts/meta_analysis.py``) and the optimization seed
for the optimizer benchmark (``scripts/benchmark_optimizers.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control, wilcoxon


def paired_stats(ref: np.ndarray, other: np.ndarray, tol: float) -> dict[str, float | int]:
    """Reference minus competitor over paired units (positive = reference better).

    Differences with ``|d| <= tol`` are ties: they count as T in W/T/L and are
    dropped from the Wilcoxon signed-rank test. Cohen's d_z = mean(d) / sd(d).
    """
    diff = np.asarray(ref, dtype=float) - np.asarray(other, dtype=float)
    wins, losses = int(np.sum(diff > tol)), int(np.sum(diff < -tol))
    nonzero = np.abs(diff) > tol
    p = float(wilcoxon(diff[nonzero]).pvalue) if nonzero.sum() > 0 else 1.0
    sd = diff.std(ddof=1) if len(diff) > 1 else np.nan
    if not sd > 0:  # undefined (n < 2) or all differences identical
        d = np.nan if len(diff) < 2 or diff.mean() != 0 else 0.0
    else:
        d = float(diff.mean() / sd)
    return {"n": len(diff), "mean_diff_vs_ref": float(diff.mean()), "median_diff_vs_ref": float(np.median(diff)),
            "wilcoxon_p": p, "cohens_dz": d, "ref_wins": wins, "ties": len(diff) - wins - losses,
            "ref_losses": losses}


def holm(pvalues: pd.Series) -> pd.Series:
    """Holm–Bonferroni adjusted p-values (family-wise error rate)."""
    order = np.argsort(pvalues.to_numpy())
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues.iloc[idx])
        adjusted[idx] = min(1.0, running)
    return pd.Series(adjusted, index=pvalues.index)


def benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    """Benjamini–Hochberg adjusted p-values (false discovery rate)."""
    if len(pvalues) == 0:
        return pvalues.astype(float)
    return pd.Series(false_discovery_control(pvalues.to_numpy(dtype=float), method="bh"), index=pvalues.index)
