"""Tests for the fixed-hyperparameter JSON configuration loader."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from prunestack.models import hyperparameters as hp
from prunestack.models.hyperparameters import load_hyperparameters, load_stacking_config
from prunestack.models.registry import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    get_model_class,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_CONFIG = REPO_ROOT / "configs" / "hyperparameters.json"
PACKAGED_CONFIG = REPO_ROOT / "src" / "prunestack" / "configs" / "hyperparameters.json"


def _write(tmp_path: Path, data: dict, name: str = "hp.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _base_config() -> dict:
    return json.loads(REPO_CONFIG.read_text(encoding="utf-8"))


class TestDefaultLoad:
    @pytest.mark.parametrize(
        "task, expected",
        [("classification", CLASSIFICATION_ALGORITHMS), ("regression", REGRESSION_ALGORITHMS)],
    )
    def test_exact_names_in_registry_order(self, task, expected):
        params = load_hyperparameters(task, seed=7)
        assert list(params.keys()) == list(expected)

    def test_iteration_limits(self):
        clf = load_hyperparameters("classification", seed=0)
        reg = load_hyperparameters("regression", seed=0)
        assert clf["LogisticRegression"]["max_iter"] == 1000
        assert clf["MLPClassifier"]["max_iter"] == 1000
        assert clf["HistGradientBoostingClassifier"]["max_iter"] == 300
        assert clf["RandomForestClassifier"]["n_estimators"] == 100
        assert reg["MLPRegressor"]["max_iter"] == 1000
        assert reg["HistGradientBoostingRegressor"]["max_iter"] == 300
        assert reg["ElasticNet"]["max_iter"] == 2000

    def test_repo_and_packaged_copies_identical(self):
        assert json.loads(REPO_CONFIG.read_text()) == json.loads(PACKAGED_CONFIG.read_text())

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_params_instantiate(self, task):
        for algo, params in load_hyperparameters(task, seed=1).items():
            get_model_class(algo, task)(**params)


class TestRandomStateInjection:
    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_only_where_supported(self, task):
        params = load_hyperparameters(task, seed=123)
        for algo, p in params.items():
            supports = "random_state" in get_model_class(algo, task)().get_params()
            if supports:
                assert p["random_state"] == 123, algo
            else:
                assert "random_state" not in p, algo

    def test_knn_has_no_random_state(self):
        assert "random_state" not in load_hyperparameters("classification", 1)["KNeighborsClassifier"]
        assert "random_state" not in load_hyperparameters("regression", 1)["KNeighborsRegressor"]

    def test_loaded_config_not_mutated(self, tmp_path):
        data = _base_config()
        original = copy.deepcopy(data)
        path = _write(tmp_path, data)
        a = load_hyperparameters("classification", seed=1, config_path=path)
        a["LogisticRegression"]["max_iter"] = -1
        b = load_hyperparameters("classification", seed=2, config_path=path)
        assert b["LogisticRegression"]["max_iter"] == 1000
        assert b["LogisticRegression"]["random_state"] == 2
        assert json.loads(path.read_text()) == original


class TestValidation:
    def test_missing_name_raises(self, tmp_path):
        data = _base_config()
        del data["classification"]["RandomForestClassifier"]
        with pytest.raises(ValueError, match="Missing: \\['RandomForestClassifier'\\]"):
            load_hyperparameters("classification", 1, _write(tmp_path, data))

    def test_unknown_name_raises(self, tmp_path):
        data = _base_config()
        data["regression"]["LinearSVR"] = {}
        with pytest.raises(ValueError, match="unknown: \\['LinearSVR'\\]"):
            load_hyperparameters("regression", 1, _write(tmp_path, data))

    def test_missing_section_raises(self, tmp_path):
        data = _base_config()
        del data["regression"]
        with pytest.raises(ValueError, match="no 'regression' section"):
            load_hyperparameters("regression", 1, _write(tmp_path, data))

    def test_metadata_keys_skipped(self, tmp_path):
        data = _base_config()
        data["classification"]["_comment"] = "ignored"
        params = load_hyperparameters("classification", 1, _write(tmp_path, data))
        assert "_comment" not in params

    def test_bad_task_type(self):
        with pytest.raises(ValueError):
            load_hyperparameters("clustering", 1)

    def test_missing_custom_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_hyperparameters("classification", 1, tmp_path / "nope.json")


class TestResolution:
    def test_custom_path(self, tmp_path):
        data = _base_config()
        data["classification"]["MLPClassifier"] = {"max_iter": 50, "hidden_layer_sizes": [10]}
        params = load_hyperparameters("classification", 3, _write(tmp_path, data))
        assert params["MLPClassifier"]["max_iter"] == 50
        assert params["MLPClassifier"]["hidden_layer_sizes"] == [10]

    def test_packaged_resource_fallback(self, monkeypatch, tmp_path):
        # Simulate a pip install without the repository-level configs/ directory.
        monkeypatch.setattr(hp, "_REPO_CONFIG_PATH", tmp_path / "missing.json")
        _, source = hp._load_raw_config(None)
        assert "package resource" in source
        params = load_hyperparameters("regression", seed=5)
        assert list(params) == list(REGRESSION_ALGORITHMS)

    def test_repo_level_preferred(self):
        _, source = hp._load_raw_config(None)
        assert source == str(REPO_CONFIG.resolve())


class TestStackingConfig:
    def test_defaults(self):
        assert load_stacking_config("classification") == ("nnls_mlr", {})
        assert load_stacking_config("regression") == ("nnls", {})

    def test_missing_section_falls_back_to_nnls(self, tmp_path):
        data = _base_config()
        del data["stacking"]
        path = _write(tmp_path, data)
        assert load_stacking_config("classification", path) == ("nnls_mlr", {})
        assert load_stacking_config("regression", path) == ("nnls", {})

    def test_custom_meta_learner(self, tmp_path):
        data = _base_config()
        data["stacking"]["regression"] = {"meta_learner": "ridge", "params": {"alpha": 0.5}}
        assert load_stacking_config("regression", _write(tmp_path, data)) == ("ridge", {"alpha": 0.5})

    def test_unknown_key_raises(self, tmp_path):
        data = _base_config()
        data["stacking"]["classification"] = {"meta_learner": "nnls_mlr", "C": 1.0}
        with pytest.raises(ValueError, match="unknown keys"):
            load_stacking_config("classification", _write(tmp_path, data))

    def test_params_must_be_object(self, tmp_path):
        data = _base_config()
        data["stacking"]["classification"] = {"meta_learner": "ridge", "params": [1]}
        with pytest.raises(ValueError, match="params"):
            load_stacking_config("classification", _write(tmp_path, data))
