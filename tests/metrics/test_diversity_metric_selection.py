"""Tests for ambiguity diversity and diversity_metric selection/propagation."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.metrics.diversity import (
    ambiguity_decomposition,
    ambiguity_diversity,
    average_diversity,
    double_fault_measure,
    pairwise_diversity_matrix,
    q_statistic,
    resolve_diversity_metric,
)
from prunestack.metrics.fitness import calculate_fitness, evaluate_subset


def _reg_data(seed=0, n=200, m=8):
    rng = np.random.RandomState(seed)
    y = rng.randn(n)
    preds = y[:, None] + rng.randn(n, m) * rng.uniform(0.2, 2.0, size=m)
    return y, preds


def _clf_data(seed=0, n=150, m=6):
    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, size=n)
    preds = np.where(rng.rand(n, m) < 0.7, y[:, None], rng.randint(0, 3, size=(n, m)))
    return y, preds.astype(float)


class TestAmbiguity:
    @pytest.mark.parametrize("seed", range(10))
    def test_identity(self, seed):
        y, preds = _reg_data(seed)
        rng = np.random.RandomState(seed + 100)
        k = rng.randint(2, preds.shape[1] + 1)
        idx = sorted(rng.choice(preds.shape[1], size=k, replace=False))
        d = ambiguity_decomposition(preds, idx, y)
        assert d["ensemble_error"] == pytest.approx(
            d["avg_member_error"] - d["ambiguity"], rel=1e-10, abs=1e-12
        )
        # direct check of E_ens
        f_bar = preds[:, idx].mean(axis=1)
        assert d["ensemble_error"] == pytest.approx(np.mean((f_bar - y) ** 2))

    @pytest.mark.parametrize("seed", range(10))
    def test_range(self, seed):
        y, preds = _reg_data(seed)
        v = ambiguity_diversity(preds, list(range(preds.shape[1])), y)
        assert 0.0 <= v <= 1.0

    def test_identical_models_zero(self):
        y, preds = _reg_data(1)
        same = np.repeat(preds[:, [0]], 5, axis=1)
        assert ambiguity_diversity(same, range(5), y) == pytest.approx(0.0, abs=1e-12)

    def test_single_model_zero(self):
        y, preds = _reg_data(2)
        assert average_diversity(preds, [3], "regression", y) == 0.0

    def test_zero_member_error(self):
        y, _ = _reg_data(3)
        perfect = np.repeat(y[:, None], 4, axis=1)
        assert ambiguity_diversity(perfect, range(4), y) == 0.0

    def test_normalised_value(self):
        y, preds = _reg_data(4)
        idx = [0, 2, 5]
        d = ambiguity_decomposition(preds, idx, y)
        assert average_diversity(preds, idx, "regression", y) == pytest.approx(
            d["ambiguity"] / d["avg_member_error"]
        )

    def test_requires_y_true(self):
        _, preds = _reg_data(5)
        with pytest.raises(ValueError):
            average_diversity(preds, [0, 1], "regression")


class TestMetricResolution:
    def test_auto(self):
        assert resolve_diversity_metric("auto", "classification") == "disagreement"
        assert resolve_diversity_metric("auto", "regression") == "ambiguity"
        assert resolve_diversity_metric(None, "regression") == "ambiguity"

    @pytest.mark.parametrize(
        "metric, task",
        [("ambiguity", "classification"), ("correlation", "classification"),
         ("disagreement", "regression"), ("q_statistic", "regression"),
         ("bogus", "classification")],
    )
    def test_invalid_combo(self, metric, task):
        with pytest.raises(ValueError):
            resolve_diversity_metric(metric, task)
        with pytest.raises(ValueError):
            average_diversity(np.zeros((5, 3)), [0, 1, 2], task, np.zeros(5),
                              diversity_metric=metric)

    def test_auto_regression_equals_ambiguity(self):
        y, preds = _reg_data(6)
        idx = list(range(preds.shape[1]))
        assert average_diversity(preds, idx, "regression", y) == average_diversity(
            preds, idx, "regression", y, diversity_metric="ambiguity")

    def test_auto_classification_equals_disagreement(self):
        y, preds = _clf_data(6)
        idx = list(range(preds.shape[1]))
        assert average_diversity(preds, idx, "classification", y) == average_diversity(
            preds, idx, "classification", y, diversity_metric="disagreement")

    def test_correlation_is_one_minus_abs_rho(self):
        y, preds = _reg_data(7)
        idx = [0, 1]
        rho = np.corrcoef(preds[:, 0], preds[:, 1])[0, 1]
        assert average_diversity(preds, idx, "regression", y,
                                 diversity_metric="correlation") == pytest.approx(1 - abs(rho))


class TestClassificationMappings:
    def test_q_statistic_mapping(self):
        y, preds = _clf_data(8)
        q = q_statistic(preds[:, 0], preds[:, 1], y)
        v = average_diversity(preds, [0, 1], "classification", y, diversity_metric="q_statistic")
        assert v == pytest.approx((1 - q) / 2)
        assert 0.0 <= v <= 1.0

    def test_double_fault_mapping(self):
        y, preds = _clf_data(9)
        df = double_fault_measure(preds[:, 0], preds[:, 1], y)
        v = average_diversity(preds, [0, 1], "classification", y, diversity_metric="double_fault")
        assert v == pytest.approx(1 - df)

    def test_pairwise_matrix_symmetric(self):
        y, preds = _clf_data(10)
        for m in ("disagreement", "q_statistic", "double_fault"):
            mat = pairwise_diversity_matrix(preds, [0, 1, 2, 3], "classification", y,
                                            diversity_metric=m)
            assert np.allclose(mat, mat.T)
            assert np.allclose(np.diag(mat), 0)

    def test_pairwise_ambiguity_rejected(self):
        y, preds = _reg_data(11)
        with pytest.raises(ValueError):
            pairwise_diversity_matrix(preds, [0, 1], "regression", y, diversity_metric="ambiguity")


class TestPropagation:
    def test_fitness_changes_with_metric(self):
        y, preds = _reg_data(12)
        idx = [0, 3, 5]
        f_auto = calculate_fitness(y, preds, idx, 0.5, "regression", performance_metric="rmse")
        f_corr = calculate_fitness(y, preds, idx, 0.5, "regression", performance_metric="rmse",
                                   diversity_metric="correlation")
        assert f_auto != pytest.approx(f_corr)
        res = evaluate_subset(y, preds, idx, 0.5, "regression", performance_metric="rmse",
                              diversity_metric="correlation")
        assert res["fitness"] == pytest.approx(f_corr)

    def test_classification_fitness_changes(self):
        y, preds = _clf_data(13)
        idx = [0, 1, 2]
        a = calculate_fitness(y, preds, idx, 0.3, "classification")
        b = calculate_fitness(y, preds, idx, 0.3, "classification", diversity_metric="double_fault")
        assert a != pytest.approx(b)

    def test_reaches_fitness_objective(self, monkeypatch):
        import prunestack.optimization.objective as obj_mod
        from prunestack.optimization.objective import FitnessObjective

        seen = []
        real = obj_mod.calculate_fitness

        def spy(*args, **kwargs):
            seen.append(kwargs.get("diversity_metric"))
            return real(*args, **kwargs)

        monkeypatch.setattr(obj_mod, "calculate_fitness", spy)
        y, preds = _reg_data(14)
        obj = FitnessObjective(y, preds, 0.5, "regression", diversity_metric="correlation")
        obj(np.array([1, 1, 0, 1, 0, 0, 0, 1]))
        assert seen == ["correlation"]

    def test_reaches_optimize_ensemble(self, monkeypatch):
        import prunestack.optimization.runner as run_mod
        from prunestack.optimization.eho import EHOOptimizer

        captured = {}
        real_cls = run_mod.FitnessObjective

        class SpyObjective(real_cls):
            def __init__(self, *a, **kw):
                captured["diversity_metric"] = kw.get("diversity_metric")
                super().__init__(*a, **kw)

        monkeypatch.setattr(run_mod, "FitnessObjective", SpyObjective)
        y, preds = _reg_data(15, n=60, m=6)
        opt = EHOOptimizer(n_clans=2, clan_size=3, max_iterations=2)
        run_mod.optimize_ensemble(y, preds, 0.5, "regression", opt, seed=1,
                                  diversity_metric="correlation")
        assert captured["diversity_metric"] == "correlation"

        df = run_mod.optimize_alpha_grid(
            y, preds, [0.2, 0.8], optimizer=opt, task_type="regression", seed=1,
            diversity_metric="rmse_difference",
        )
        assert captured["diversity_metric"] == "rmse_difference"
        assert len(df) == 2

    def test_config_accepts_new_metrics(self, tmp_path):
        from prunestack.config import ExperimentConfig, DEFAULTS
        assert DEFAULTS["diversity_metric"] == "auto"
        for m in ("auto", "ambiguity"):
            assert ExperimentConfig(dataset_path=tmp_path, diversity_metric=m).diversity_metric == m
        with pytest.raises(ValueError):
            ExperimentConfig(dataset_path=tmp_path, diversity_metric="nope")

    def test_cli_options(self, tmp_path):
        from prunestack.cli import parse_args
        hp_file = tmp_path / "hp.json"
        hp_file.write_text("{}")
        cfg, _ = parse_args([
            "--dataset-path", str(tmp_path), "--diversity-metric", "correlation",
            "--hyperparameters", str(hp_file),
        ])
        assert cfg.diversity_metric == "correlation"
        assert cfg.hyperparameters_path == hp_file

    def test_config_missing_hyperparameters_file(self, tmp_path):
        from prunestack.config import ExperimentConfig
        with pytest.raises(FileNotFoundError):
            ExperimentConfig(dataset_path=tmp_path, hyperparameters_path=tmp_path / "x.json")
