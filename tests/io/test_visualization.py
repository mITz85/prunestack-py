"""Tests for prunestack.io.visualization."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from prunestack.io.results import ExperimentResults
from prunestack.io.visualization import (
    generate_experiment_report,
    plot_alpha_tradeoff,
    plot_ensemble_size_vs_fitness,
    plot_optimization_convergence,
    plot_strategy_comparison,
)

pytest.importorskip("matplotlib")


@pytest.fixture
def optimization_df():
    return pd.DataFrame({
        "alpha": [0.0, 0.5, 1.0],
        "best_fitness": [0.4, 0.6, 0.8],
        "n_models_selected": [30, 12, 5],
        "n_iterations": [10, 10, 8],
        "execution_time": [1.0, 1.1, 0.9],
    })


@pytest.fixture
def comparison_df():
    rows = []
    for alpha in (0.0, 1.0):
        for name, perf in [("full_ensemble", 0.7), ("optimized_ensemble", 0.75),
                           ("full_stacked", 0.72), ("optimized_stacked", 0.78)]:
            rows.append({"strategy_name": name, "alpha": alpha, "n_models": 10,
                         "performance": perf, "diversity": 0.3, "fitness": 0.5})
    return pd.DataFrame(rows)


def test_convergence_plot(tmp_path):
    convergence = [{"alpha": 0.0, "convergence_history": [0.1, 0.2, 0.3]},
                   {"alpha": 1.0, "convergence_history": [0.5, 0.6]}]
    path = plot_optimization_convergence(convergence, tmp_path / "convergence.png")
    assert path.exists() and path.stat().st_size > 0


def test_alpha_tradeoff_plot(tmp_path, optimization_df):
    assert plot_alpha_tradeoff(optimization_df, tmp_path / "tradeoff.png").exists()


def test_size_vs_fitness_plot(tmp_path, optimization_df):
    assert plot_ensemble_size_vs_fitness(optimization_df, tmp_path / "size.png").exists()


def test_strategy_comparison_plot(tmp_path, comparison_df):
    assert plot_strategy_comparison(comparison_df, tmp_path / "strategies.png").exists()


def test_missing_columns_returns_none(tmp_path):
    assert plot_alpha_tradeoff(pd.DataFrame({"alpha": [0.0]}), tmp_path / "x.png") is None


def test_report_with_all_sections(tmp_path, optimization_df, comparison_df):
    (tmp_path / "optimization").mkdir()
    (tmp_path / "optimization" / "optimization_convergence.json").write_text(
        json.dumps([{"alpha": 0.5, "convergence_history": [0.1, 0.4]}])
    )
    results = ExperimentResults(
        dataset_info={"task_type": "classification", "n_samples": 100, "dataset_path": "<data>"},
        optimization_results=optimization_df,
        strategy_comparison=comparison_df,
        meta_features={"n_samples": 100.0},
        execution_metadata={"package_version": "2.0.0"},
    )
    report = generate_experiment_report(results, tmp_path / "experiment_report.html")
    text = report.read_text(encoding="utf-8")
    assert "&lt;data&gt;" in text  # escaped
    assert "2.0.0" in text
    for name in ("alpha_tradeoff.png", "size_vs_fitness.png", "convergence.png", "strategy_comparison.png"):
        assert (tmp_path / "plots" / name).exists(), name
        assert f"plots/{name}" in text


def test_report_without_optimization(tmp_path):
    results = ExperimentResults(dataset_info={"task_type": "regression"})
    report = generate_experiment_report(results, tmp_path / "report.html")
    assert report.exists()
    assert not (tmp_path / "plots" / "convergence.png").exists()
