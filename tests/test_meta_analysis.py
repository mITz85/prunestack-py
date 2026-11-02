"""Tests for prunestack.analysis.stats and scripts/meta_analysis.py on synthetic results."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prunestack.analysis.stats import benjamini_hochberg, holm, paired_stats

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "meta_analysis.py"
ALPHAS = np.round(np.linspace(0, 1, 11), 1)


def load_script():
    spec = importlib.util.spec_from_file_location("meta_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStats:
    def test_paired_stats_wtl_and_ties(self):
        s = paired_stats(np.array([0.9, 0.8, 0.7, 0.5]), np.array([0.8, 0.8, 0.75, 0.4]), tol=1e-3)
        assert (s["ref_wins"], s["ties"], s["ref_losses"]) == (2, 1, 1)
        assert s["n"] == 4 and 0 <= s["wilcoxon_p"] <= 1

    def test_holm_and_bh(self):
        p = pd.Series([0.01, 0.04, 0.03])
        assert holm(p).tolist() == pytest.approx([0.03, 0.06, 0.06])
        assert benjamini_hochberg(p).tolist() == pytest.approx([0.03, 0.04, 0.04])


def write_fake_results(root: Path, n_per_task: int = 6) -> None:
    rng = np.random.default_rng(0)
    for i in range(2 * n_per_task):
        task = "classification" if i < n_per_task else "regression"
        signal = rng.uniform()
        alpha_star = float(np.round(signal, 1))
        res = root / f"d{i}" / "results"
        res.mkdir(parents=True)
        curve = 0.7 - 0.2 * (ALPHAS - alpha_star) ** 2
        pd.DataFrame({"alpha": ALPHAS, "pruned_vote_mean": curve, "pruned_stacked_mean": curve + 0.01}).to_csv(
            res / "alpha_curve.csv", index=False)
        target = {"alpha_star": alpha_star, "score_at_alpha_star": 0.7, "curve_range": 0.2, "gain_over_alpha_1": 0.0}
        (res / "alpha_star.json").write_text(json.dumps({
            "task_type": task, "n_runs": 3,
            "reference_scores": {"full_vote": 0.65, "full_stacked": 0.68, "best_single": 0.6},
            "pruned_vote": target, "pruned_stacked": target}))
        (res / "meta_features.json").write_text(json.dumps({
            "informative": signal, "copy": 2 * signal + 1, "constant": 1.0, "noise": float(rng.normal())}))
        (res / "summary.json").write_text(json.dumps({"protocol": "nested", "n_samples": 100, "n_features": 5}))


def test_meta_analysis_end_to_end(tmp_path):
    write_fake_results(tmp_path / "data")
    out = tmp_path / "out"
    assert load_script().main(["--data-root", str(tmp_path / "data"), "--output-dir", str(out)]) == 0

    filt = pd.read_csv(out / "meta_feature_filter.csv")
    reasons = filt[filt.group == "classification"].set_index("meta_feature")["reason"]
    assert reasons["constant"] == "constant" and reasons["copy"].startswith("|rho|=1.00 with informative")

    corr = pd.read_csv(out / "meta_feature_correlations.csv")
    top = corr[(corr.group == "all") & (corr.target == "pruned_vote")].iloc[0]
    assert top.meta_feature == "informative" and top.p_bh < 0.05

    loo = pd.read_csv(out / "loo_alpha_predictions.csv")
    assert (loo["regret_alpha_median"] >= -1e-12).all()  # oracle alpha* is the curve maximum
    comp = pd.read_csv(out / "strategy_comparison.csv")
    assert set(comp.group) == {"classification", "regression", "all"}
    assert {"wilcoxon_p_holm", "cohens_dz", "ref_wins", "ties", "ref_losses"} <= set(comp.columns)
