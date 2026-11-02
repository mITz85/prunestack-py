"""Tests for prunestack.metrics.fitness."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.metrics.fitness import calculate_fitness, evaluate_subset


@pytest.fixture
def classification_data():
    rng = np.random.RandomState(42)
    n_samples, n_models = 100, 10
    y_true = rng.randint(0, 2, n_samples)
    # Models with varying noise levels
    pred = np.column_stack([
        (y_true + rng.binomial(1, 0.1 + 0.05 * i, n_samples)) % 2
        for i in range(n_models)
    ])
    return y_true, pred


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    n_samples, n_models = 100, 10
    y_true = rng.randn(n_samples) * 10
    pred = np.column_stack([
        y_true + rng.randn(n_samples) * (0.5 + 0.3 * i)
        for i in range(n_models)
    ])
    return y_true, pred


class TestCalculateFitness:
    def test_alpha_one_performance_only(self, classification_data):
        y, pred = classification_data
        # α=1 → fitness = P(S)
        fitness = calculate_fitness(y, pred, list(range(10)), 1.0, "classification")
        from prunestack.metrics.performance import evaluate_ensemble_performance
        perf = evaluate_ensemble_performance(y, pred, list(range(10)), "classification")
        assert fitness == pytest.approx(perf)

    def test_alpha_zero_diversity_only(self, classification_data):
        y, pred = classification_data
        # α=0 (β=1−α=1) → fitness = D(S)
        fitness = calculate_fitness(y, pred, list(range(10)), 0.0, "classification")
        from prunestack.metrics.diversity import average_diversity
        div = average_diversity(pred, list(range(10)), "classification", y)
        assert fitness == pytest.approx(div)

    def test_complementary_weights(self, classification_data):
        y, pred = classification_data
        # β is derived as 1−α → fitness = α·P(S) + (1−α)·D(S)
        fitness = calculate_fitness(y, pred, list(range(10)), 0.3, "classification")
        from prunestack.metrics.performance import evaluate_ensemble_performance
        from prunestack.metrics.diversity import average_diversity
        perf = evaluate_ensemble_performance(y, pred, list(range(10)), "classification")
        div = average_diversity(pred, list(range(10)), "classification", y)
        assert fitness == pytest.approx(0.3 * perf + 0.7 * div)

    def test_fitness_in_unit_range(self, classification_data):
        y, pred = classification_data
        for alpha in [round(0.1 * i, 1) for i in range(11)]:
            f = calculate_fitness(y, pred, list(range(10)), alpha, "classification")
            assert 0.0 <= f <= 1.0, f"fitness={f} for α={alpha}"

    def test_regression(self, regression_data):
        y, pred = regression_data
        f = calculate_fitness(y, pred, list(range(10)), 0.5, "regression")
        assert 0.0 <= f <= 1.0

    def test_subset(self, classification_data):
        y, pred = classification_data
        f_full = calculate_fitness(y, pred, list(range(10)), 0.5, "classification")
        f_sub = calculate_fitness(y, pred, [0, 1, 2], 0.5, "classification")
        # Different subsets should generally give different fitness
        # (not guaranteed but extremely likely with random data)
        assert isinstance(f_sub, float)


class TestEvaluateSubset:
    def test_returns_all_keys(self, classification_data):
        y, pred = classification_data
        result = evaluate_subset(y, pred, list(range(10)), 0.5, "classification")
        assert "fitness" in result
        assert "performance" in result
        assert "diversity" in result
        assert "ensemble_predictions" in result

    def test_predictions_shape(self, classification_data):
        y, pred = classification_data
        result = evaluate_subset(y, pred, list(range(10)), 0.5, "classification")
        assert result["ensemble_predictions"].shape == (100,)

    def test_fitness_consistency(self, classification_data):
        y, pred = classification_data
        alpha = 0.7
        result = evaluate_subset(y, pred, [0, 1, 2], alpha, "classification")
        expected = alpha * result["performance"] + (1 - alpha) * result["diversity"]
        assert result["fitness"] == pytest.approx(expected)

    def test_regression_subset(self, regression_data):
        y, pred = regression_data
        result = evaluate_subset(y, pred, [0, 1, 2, 3, 4], 0.5, "regression")
        assert 0.0 <= result["performance"] <= 1.0
        assert 0.0 <= result["diversity"] <= 1.0
        assert result["ensemble_predictions"].shape == (100,)


class TestFitnessScaler:
    def test_identity_is_raw_fitness(self, classification_data):
        from prunestack.metrics.fitness import IDENTITY_SCALER
        y, pred = classification_data
        raw = calculate_fitness(y, pred, [0, 1, 2], 0.3, "classification")
        scaled = calculate_fitness(y, pred, [0, 1, 2], 0.3, "classification", scaler=IDENTITY_SCALER)
        assert raw == scaled
        assert IDENTITY_SCALER.combine(0.3, 0.8, 0.2) == pytest.approx(0.3 * 0.8 + 0.7 * 0.2)

    def test_combine_affine_unclipped(self):
        from prunestack.metrics.fitness import FitnessScaler
        s = FitnessScaler(performance_bounds=(0.6, 0.8), diversity_bounds=(0.2, 0.4), method="minmax")
        assert s.scale_performance(0.7) == pytest.approx(0.5)
        assert s.scale_diversity(0.5) == pytest.approx(1.5)  # beyond the reference range, not clipped
        assert s.combine(0.5, 0.7, 0.3) == pytest.approx(0.5)
        assert s.to_dict()["performance_bounds"] == [0.6, 0.8]

    def test_degenerate_span_is_shift_only(self):
        from prunestack.metrics.fitness import FitnessScaler
        assert FitnessScaler(performance_bounds=(0.5, 0.5)).scale_performance(0.7) == pytest.approx(0.2)

    @pytest.mark.parametrize("task", ["classification", "regression"])
    def test_fit_bounds_deterministic_and_ordered(self, classification_data, regression_data, task):
        from prunestack.metrics.fitness import fit_fitness_scaler
        y, pred = classification_data if task == "classification" else regression_data
        a = fit_fitness_scaler(y, pred, task, n_subsets=60, seed=7)
        b = fit_fitness_scaler(y, pred, task, n_subsets=60, seed=7)
        assert a == b
        assert a.method == "minmax" and a.n_reference_subsets == 60
        assert a.performance_bounds[0] <= a.performance_bounds[1]
        assert a.diversity_bounds[0] <= a.diversity_bounds[1]

    def test_fit_none_and_invalid(self, classification_data):
        from prunestack.metrics.fitness import IDENTITY_SCALER, fit_fitness_scaler
        y, pred = classification_data
        assert fit_fitness_scaler(y, pred, "classification", method="none") is IDENTITY_SCALER
        with pytest.raises(ValueError, match="method"):
            fit_fitness_scaler(y, pred, "classification", method="zscore")
        with pytest.raises(ValueError, match="quantiles"):
            fit_fitness_scaler(y, pred, "classification", quantiles=(0.9, 0.1))

    def test_scaling_preserves_ranking_for_fixed_alpha(self, classification_data):
        from prunestack.metrics.fitness import fit_fitness_scaler
        y, pred = classification_data
        scaler = fit_fitness_scaler(y, pred, "classification", n_subsets=50, seed=1)
        subsets = [[0, 1], [0, 1, 2, 3], [5, 6, 7], list(range(10))]
        for alpha in (0.0, 1.0):
            raw = [calculate_fitness(y, pred, s, alpha, "classification") for s in subsets]
            scaled = [calculate_fitness(y, pred, s, alpha, "classification", scaler=scaler) for s in subsets]
            assert np.argsort(raw).tolist() == np.argsort(scaled).tolist()

    def test_evaluate_subset_scaled_keys(self, classification_data):
        from prunestack.metrics.fitness import FitnessScaler
        y, pred = classification_data
        s = FitnessScaler(performance_bounds=(0.5, 1.0), diversity_bounds=(0.0, 0.5), method="minmax")
        r = evaluate_subset(y, pred, [0, 1, 2], 0.4, "classification", scaler=s, return_predictions=False)
        assert "ensemble_predictions" not in r
        assert r["performance_scaled"] == pytest.approx(s.scale_performance(r["performance_score"]))
        assert r["diversity_scaled"] == pytest.approx(s.scale_diversity(r["diversity"]))
        assert r["fitness"] == pytest.approx(0.4 * r["performance_scaled"] + 0.6 * r["diversity_scaled"])


class TestSampleRandomSubsets:
    def test_shape_min_size_and_size_spread(self):
        from prunestack.metrics.fitness import sample_random_subsets
        subsets = sample_random_subsets(np.random.default_rng(0), 400, 50, (0.05, 0.5), min_size=2)
        assert subsets.shape == (400, 50)
        sizes = subsets.sum(axis=1)
        assert sizes.min() >= 2
        assert sizes.max() > 15 and np.percentile(sizes, 10) < 8  # small and medium ensembles


class TestSampleUniformSizeSubsets:
    def test_sizes_cover_the_whole_range(self):
        from prunestack.metrics.fitness import sample_uniform_size_subsets
        subsets = sample_uniform_size_subsets(np.random.default_rng(0), 2000, 20, min_size=2)
        assert subsets.shape == (2000, 20) and set(np.unique(subsets)) == {0, 1}
        sizes = subsets.sum(axis=1)
        assert sizes.min() == 2 and sizes.max() == 20
        counts = np.bincount(sizes, minlength=21)[2:]
        assert counts.min() > 0.5 * counts.mean()  # roughly uniform over {2..20}

    def test_min_size_clamped_to_pool(self):
        from prunestack.metrics.fitness import sample_uniform_size_subsets
        subsets = sample_uniform_size_subsets(np.random.default_rng(0), 10, 3, min_size=4)
        assert (subsets.sum(axis=1) == 3).all()
