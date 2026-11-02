"""End-to-end tests for prunestack.experiment.runner on small synthetic datasets."""

from __future__ import annotations

import json

import pytest

from prunestack.config import ExperimentConfig
from prunestack.experiment.runner import ExperimentRunner

STRATEGIES = {"full_ensemble", "optimized_ensemble", "full_stacked", "optimized_stacked"}


def small_config(dataset_dir, **overrides) -> ExperimentConfig:
    settings = dict(
        dataset_path=dataset_dir,
        seed=42,
        n_folds=3,
        max_models_per_algo=3,
        eho_n_clans=2,
        eho_clan_size=4,
        eho_max_iterations=3,
        alpha_grid=[0.0, 1.0],
    )
    settings.update(overrides)
    return ExperimentConfig(**settings)


@pytest.mark.timeout(300)
class TestExperimentRunner:
    def test_classification_pipeline(self, classification_dataset_dir, tmp_path):
        out = tmp_path / "classification"
        results = ExperimentRunner(small_config(classification_dataset_dir), output_dir=out).run_full_experiment()

        assert results.dataset_info["task_type"] == "classification"
        assert results.dataset_info["n_models"] == 6 * 3
        assert list(results.optimization_results["alpha"]) == [0.0, 1.0]
        assert set(results.strategy_comparison["strategy_name"]) == STRATEGIES
        assert results.stacking_results is not None
        assert results.meta_features
        assert results.execution_metadata["config"]["seed"] == 42

        for relative in [
            "meta_features.json",
            "experiment_summary.json",
            "experiment_report.html",
            "optimization/optimization_convergence.json",
            "optimization/fitness_normalization.json",
        ]:
            assert (out / relative).exists(), relative
        scaler = json.loads((out / "optimization/fitness_normalization.json").read_text())
        assert scaler["method"] == "minmax"
        assert scaler["performance_bounds"][0] < scaler["performance_bounds"][1]
        assert results.execution_metadata["fitness_normalization"]["method"] == "minmax"

    def test_regression_pipeline(self, regression_dataset_dir, tmp_path):
        out = tmp_path / "regression"
        results = ExperimentRunner(small_config(regression_dataset_dir), output_dir=out).run_full_experiment()
        assert results.dataset_info["task_type"] == "regression"
        assert set(results.strategy_comparison["strategy_name"]) == STRATEGIES

    def test_stacking_method_override(self, classification_dataset_dir, tmp_path):
        config = small_config(classification_dataset_dir, stacking_method="logistic")
        runner = ExperimentRunner(config, skip_optimization=True, output_dir=tmp_path)
        runner.run_full_experiment()
        summary = json.loads((tmp_path / "experiment_summary.json").read_text())
        assert "logistic" in json.dumps(summary)

    def test_skip_optimization(self, classification_dataset_dir, tmp_path):
        runner = ExperimentRunner(
            small_config(classification_dataset_dir), skip_optimization=True, output_dir=tmp_path,
        )
        results = runner.run_full_experiment()
        assert results.optimization_results is None
        assert set(results.strategy_comparison["alpha"]) == {0.5}

    def test_skip_stacking(self, classification_dataset_dir, tmp_path):
        runner = ExperimentRunner(
            small_config(classification_dataset_dir), skip_stacking=True, output_dir=tmp_path,
        )
        results = runner.run_full_experiment()
        assert results.stacking_results is None
        assert results.strategy_comparison is None
        assert not (tmp_path / "checkpoint.json").exists()


@pytest.mark.timeout(300)
class TestRepeatedRuns:
    def test_base_models_trained_once_and_seeds_per_run(self, classification_dataset_dir, tmp_path, monkeypatch):
        import prunestack.experiment.runner as runner_module

        calls = []
        original = runner_module.train_base_models
        monkeypatch.setattr(runner_module, "train_base_models",
                            lambda *a, **k: calls.append(1) or original(*a, **k))
        runner = ExperimentRunner(small_config(classification_dataset_dir), output_dir=tmp_path)
        results = runner.run_repeated_experiment(3, first_seed=5)

        assert len(calls) == 1
        assert len(results) == 3
        assert (tmp_path / "base_models" / "base_models_predictions.csv").exists()
        for run, seed in zip((1, 2, 3), (5, 6, 7)):
            summary = json.loads((tmp_path / f"run_{run:02d}" / "experiment_summary.json").read_text())
            assert summary["execution_metadata"]["config"]["optimization_seed"] == seed
            assert (tmp_path / f"run_{run:02d}" / "optimization" / "optimization_results.csv").exists()
        # The runner is restored after the loop.
        assert runner.output_dir == tmp_path and runner.config.optimization_seed is None

    def test_single_repeated_run_equals_full_experiment(self, regression_dataset_dir, tmp_path):
        config = small_config(regression_dataset_dir, optimization_seed=1)
        full = ExperimentRunner(config, output_dir=tmp_path / "full").run_full_experiment()
        rep = ExperimentRunner(small_config(regression_dataset_dir), output_dir=tmp_path / "rep",
                               n_jobs=2).run_repeated_experiment(1, first_seed=1)[0]
        cols = ["alpha", "best_fitness", "best_subset_indices"]
        assert full.optimization_results[cols].equals(rep.optimization_results[cols])

    def test_invalid_arguments(self, classification_dataset_dir, tmp_path):
        with pytest.raises(ValueError):
            ExperimentRunner(small_config(classification_dataset_dir), output_dir=tmp_path, n_jobs=0)
        runner = ExperimentRunner(small_config(classification_dataset_dir), output_dir=tmp_path)
        with pytest.raises(ValueError):
            runner.run_repeated_experiment(0)
