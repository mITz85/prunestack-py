"""Tests for the nested evaluation protocol (prunestack.experiment.nested)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from prunestack.config import ExperimentConfig
from prunestack.experiment.nested import NestedExperiment, alpha_targets, inner_folds

ALPHAS = [0.0, 0.5, 1.0]


def nested_config(dataset_dir, **overrides) -> ExperimentConfig:
    settings = dict(
        dataset_path=dataset_dir, seed=42, n_folds=3, n_outer_folds=3, max_models_per_algo=2,
        eho_n_clans=2, eho_clan_size=3, eho_max_iterations=3, alpha_grid=ALPHAS,
        fitness_reference_subsets=50,
    )
    settings.update(overrides)
    return ExperimentConfig(**settings)


class TestAlphaTargets:
    def test_argmax_ties_and_1se(self):
        alphas = np.array([0.0, 0.5, 1.0])
        scores = np.array([[0.70, 0.80, 0.80], [0.72, 0.80, 0.78]])
        t = alpha_targets(alphas, scores)
        assert t["alpha_star"] == 0.5
        assert t["alpha_star_per_run"] == [0.5, 0.5]
        assert t["score_at_alpha_star"] == pytest.approx(0.80)
        assert t["gain_over_alpha_1"] == pytest.approx(0.01)
        assert 0.5 in t["alphas_within_1se"] and 0.0 not in t["alphas_within_1se"]
        assert t["curve_range"] == pytest.approx(0.09)

    def test_single_run(self):
        t = alpha_targets(np.array([0.0, 1.0]), np.array([[0.6, 0.5]]))
        assert t["alpha_star"] == 0.0
        assert t["alphas_within_1se"] == [0.0]


def test_inner_folds_partition_training_rows():
    y = np.array([0, 1] * 15)
    folds = inner_folds(y, "classification", n_folds=5, seed=1)
    assert len(folds) == 5
    assert sorted(np.concatenate([te for _, te in folds]).tolist()) == list(range(30))


@pytest.mark.timeout(600)
@pytest.mark.parametrize("fixture", ["classification_dataset_dir", "regression_dataset_dir"])
def test_nested_pipeline(fixture, request, tmp_path):
    dataset_dir = request.getfixturevalue(fixture)
    out = tmp_path / "nested"
    targets = NestedExperiment(nested_config(dataset_dir), output_dir=out).run(n_runs=2)

    for name in ["selection.csv", "heldout_performance.csv", "alpha_curve.csv", "alpha_star.json",
                 "meta_features.json", "fitness_normalization.json", "summary.json",
                 "base_models/heldout_predictions.csv", "base_models/model_pool.json"]:
        assert (out / name).exists(), name

    selection = pd.read_csv(out / "selection.csv")
    assert len(selection) == 2 * 3 * len(ALPHAS)  # runs x outer folds x alphas
    assert set(selection["fold"]) == {0, 1, 2}
    assert (selection["n_models_selected"] >= 1).all()

    heldout = pd.read_csv(out / "heldout_performance.csv")
    assert {"full_vote", "full_stacked", "best_single", "pruned_vote", "pruned_stacked"} <= set(heldout["strategy"])
    assert heldout["score"].between(-np.inf, 1.0).all()

    # Every row has exactly one held-out prediction of every model.
    preds = pd.read_csv(out / "base_models/heldout_predictions.csv")
    assert not preds.isna().any().any()
    assert preds.shape[1] == 6 * 2

    assert targets["pruned_vote"]["alpha_star"] in ALPHAS
    assert json.loads((out / "alpha_star.json").read_text()) == json.loads(json.dumps(targets))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["protocol"] == "nested" and summary["optimization_seeds"] == [1, 2]

    # A second run resumes from the fold checkpoints and reproduces the results.
    again = NestedExperiment(nested_config(dataset_dir), output_dir=out).run(n_runs=2)
    assert again == targets


def test_heldout_rows_never_used_for_selection(classification_dataset_dir, tmp_path, monkeypatch):
    """The optimizer only receives inner predictions of the outer training rows."""
    import prunestack.experiment.nested as nested

    seen: list[int] = []
    original = nested.optimize_alpha_grid

    def spy(y_true, predictions, *args, **kwargs):
        seen.append(len(y_true))
        assert predictions.shape[0] == len(y_true)
        return original(y_true, predictions, *args, **kwargs)

    monkeypatch.setattr(nested, "optimize_alpha_grid", spy)
    exp = NestedExperiment(nested_config(classification_dataset_dir), output_dir=tmp_path / "o")
    exp.run(n_runs=1)
    train_sizes = sorted(len(tr) for tr, _ in exp.bundle.folds)
    assert sorted(seen) == train_sizes
