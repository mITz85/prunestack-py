"""Tests for prunestack.io.results."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prunestack.io.results import ExperimentResults, save_results, load_results


@pytest.fixture
def sample_results():
    return ExperimentResults(
        dataset_info={"task_type": "classification", "n_samples": 100},
        base_models_performance=pd.DataFrame({
            "model_id": [0, 1], "metric_value": [0.8, 0.75]
        }),
        optimization_results=pd.DataFrame({
            "alpha": [0.5], "beta": [0.5], "best_fitness": [0.7]
        }),
        stacking_results=pd.DataFrame({
            "subset": ["full"], "stacking_performance": [0.85]
        }),
        strategy_comparison=pd.DataFrame({
            "strategy_name": ["full_ensemble"], "performance": [0.8]
        }),
        meta_features={"n_samples": 100.0, "n_features": 10.0},
        execution_metadata={"seed": 42, "total_time": 5.0},
    )


class TestExperimentResults:

    def test_default_init(self):
        r = ExperimentResults()
        assert r.dataset_info == {}
        assert r.base_models_performance is None
        assert r.meta_features == {}

    def test_init_with_data(self, sample_results):
        assert sample_results.dataset_info["task_type"] == "classification"
        assert sample_results.base_models_performance is not None
        assert len(sample_results.meta_features) == 2


class TestSaveLoad:

    def test_save_creates_directories(self, sample_results, tmp_path):
        output = tmp_path / "test_output"
        save_results(sample_results, output)
        assert (output / "base_models").is_dir()
        assert (output / "optimization").is_dir()
        assert (output / "stacking").is_dir()
        assert (output / "comparison").is_dir()

    def test_save_creates_files(self, sample_results, tmp_path):
        output = tmp_path / "test_output"
        save_results(sample_results, output)
        assert (output / "base_models" / "base_models_performance.csv").exists()
        assert (output / "optimization" / "optimization_results.csv").exists()
        assert (output / "stacking" / "stacking_results.csv").exists()
        assert (output / "comparison" / "strategy_comparison.csv").exists()
        assert (output / "meta_features.json").exists()
        assert (output / "experiment_summary.json").exists()

    def test_meta_features_json_valid(self, sample_results, tmp_path):
        output = tmp_path / "test_output"
        save_results(sample_results, output)
        with open(output / "meta_features.json") as f:
            mf = json.load(f)
        assert mf["n_samples"] == 100.0
        assert mf["n_features"] == 10.0

    def test_round_trip(self, sample_results, tmp_path):
        output = tmp_path / "test_output"
        save_results(sample_results, output)
        loaded = load_results(output)
        assert loaded.dataset_info["task_type"] == "classification"
        assert len(loaded.meta_features) == 2
        assert loaded.base_models_performance is not None
        assert loaded.optimization_results is not None
        assert loaded.stacking_results is not None
        assert loaded.strategy_comparison is not None

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nonexistent")

    def test_save_with_numpy_types(self, tmp_path):
        results = ExperimentResults(
            meta_features={"val": np.float64(3.14), "arr": np.array([1, 2])},
        )
        output = tmp_path / "numpy_test"
        save_results(results, output)
        with open(output / "meta_features.json") as f:
            data = json.load(f)
        assert data["val"] == 3.14
