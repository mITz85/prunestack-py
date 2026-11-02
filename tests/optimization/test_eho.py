"""Tests for prunestack.optimization.eho."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.optimization.eho import EHOOptimizer, _sigmoid, _to_binary, _ensure_valid
from prunestack.optimization.protocol import OptimizationResult


class TestSigmoid:
    def test_zero(self):
        assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(np.array([100.0]))[0] == pytest.approx(1.0)

    def test_large_negative(self):
        assert _sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0)

    def test_vectorized(self):
        x = np.array([-10, 0, 10])
        s = _sigmoid(x)
        assert s[0] < 0.01
        assert s[1] == pytest.approx(0.5)
        assert s[2] > 0.99


class TestToBinary:
    def test_output_is_binary(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(20)
        binary = _to_binary(x, rng)
        assert set(binary).issubset({0.0, 1.0})
        assert len(binary) == 20


class TestEnsureValid:
    def test_empty_gets_one(self):
        rng = np.random.default_rng(42)
        binary = np.zeros(10)
        result = _ensure_valid(binary, rng)
        assert np.sum(result) >= 1

    def test_nonempty_unchanged(self):
        rng = np.random.default_rng(42)
        binary = np.array([1, 0, 1, 0, 0])
        result = _ensure_valid(binary.copy(), rng)
        assert np.sum(result) == 2


class TestEHOOptimizer:
    """Tests for EHO optimizer."""

    def test_initialization_defaults(self):
        opt = EHOOptimizer()
        assert opt.variant == "standard"
        assert opt.n_clans == 10
        assert opt.clan_size == 10
        assert opt.max_iterations == 100
        assert opt.early_stop_patience == 25
        assert opt.alpha == 0.5
        assert opt.beta == 0.1
        assert opt.init_inclusion == (0.05, 0.5)
        assert opt.mutation_rate is None
        assert opt.max_evaluations is None
        assert opt.population_size == 100

    def test_initialization_custom(self):
        opt = EHOOptimizer(n_clans=3, clan_size=5, max_iterations=20)
        assert opt.n_clans == 3
        assert opt.clan_size == 5
        assert opt.max_iterations == 20
        assert opt.population_size == 15

    def test_invalid_n_clans(self):
        with pytest.raises(ValueError, match="n_clans"):
            EHOOptimizer(n_clans=0)

    def test_invalid_clan_size(self):
        with pytest.raises(ValueError, match="clan_size"):
            EHOOptimizer(clan_size=1)

    def test_invalid_max_iterations(self):
        with pytest.raises(ValueError, match="max_iterations"):
            EHOOptimizer(max_iterations=0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            EHOOptimizer(alpha=1.5)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta"):
            EHOOptimizer(beta=-0.1)

    def test_simple_maximize_sum(self):
        """Optimizer should find subset that maximizes sum of binary vector."""
        def objective(binary):
            return float(np.sum(binary))

        opt = EHOOptimizer(
            n_clans=2, clan_size=5, max_iterations=20,
            early_stop_patience=0,
        )
        result = opt.optimize(objective, n_models=5, seed=42)

        assert isinstance(result, OptimizationResult)
        assert result.best_fitness > 0
        assert result.n_selected >= 1
        assert len(result.convergence_history) > 0
        assert result.execution_time > 0

    def test_convergence_tracking(self):
        """Convergence history should be non-decreasing (best-so-far)."""
        def objective(binary):
            return float(np.sum(binary))

        opt = EHOOptimizer(
            n_clans=2, clan_size=5, max_iterations=15,
            early_stop_patience=0,
        )
        result = opt.optimize(objective, n_models=10, seed=42)

        # Best-so-far should be non-decreasing
        for i in range(1, len(result.convergence_history)):
            assert result.convergence_history[i] >= result.convergence_history[i - 1] - 1e-10

    def test_deterministic_seeding(self):
        """Same seed should produce the same result."""
        def objective(binary):
            return float(np.sum(binary) + np.sum(binary[:3]) * 0.5)

        opt = EHOOptimizer(
            n_clans=2, clan_size=5, max_iterations=10,
            early_stop_patience=0,
        )

        result1 = opt.optimize(objective, n_models=8, seed=123)
        result2 = opt.optimize(objective, n_models=8, seed=123)

        assert result1.best_fitness == result2.best_fitness
        np.testing.assert_array_equal(result1.best_subset, result2.best_subset)
        assert result1.convergence_history == result2.convergence_history

    def test_different_seeds_different_results(self):
        """Different seeds should (almost certainly) produce different results."""
        def objective(binary):
            # Weighted sum so there's a unique optimum
            weights = np.arange(1, 11, dtype=float)
            return float(np.dot(binary, weights))

        opt = EHOOptimizer(
            n_clans=2, clan_size=5, max_iterations=10,
            early_stop_patience=0,
        )

        result1 = opt.optimize(objective, n_models=10, seed=42)
        result2 = opt.optimize(objective, n_models=10, seed=999)

        # Different seeds may occasionally produce same result,
        # but convergence paths should differ
        histories_differ = result1.convergence_history != result2.convergence_history
        subsets_differ = not np.array_equal(result1.best_subset, result2.best_subset)
        assert histories_differ or subsets_differ

    def test_binary_output(self):
        """Best subset should be a binary vector."""
        def objective(binary):
            return float(np.sum(binary))

        opt = EHOOptimizer(n_clans=2, clan_size=3, max_iterations=5,
                           early_stop_patience=0)
        result = opt.optimize(objective, n_models=10, seed=42)

        assert set(result.best_subset.tolist()).issubset({0.0, 1.0})

    def test_early_stopping(self):
        """Test that early stopping works."""
        call_count = 0

        def constant_objective(binary):
            nonlocal call_count
            call_count += 1
            return 1.0  # constant = no improvement

        opt = EHOOptimizer(
            n_clans=2, clan_size=3, max_iterations=100,
            early_stop_patience=5,
        )
        result = opt.optimize(constant_objective, n_models=5, seed=42)

        # Should stop well before 100 iterations
        assert result.n_iterations < 100
        assert result.metadata.get("early_stopped") is True

    def test_metadata(self):
        def objective(binary):
            return float(np.sum(binary))

        opt = EHOOptimizer(n_clans=3, clan_size=4, max_iterations=5,
                           early_stop_patience=0)
        result = opt.optimize(objective, n_models=5, seed=42)

        assert "n_evaluations" in result.metadata
        assert "n_clans" in result.metadata
        assert result.metadata["n_clans"] == 3
        assert result.metadata["clan_size"] == 4

    def test_repr(self):
        opt = EHOOptimizer(n_clans=3, clan_size=5)
        s = repr(opt)
        assert "EHOOptimizer" in s
        assert "n_clans=3" in s


class TestEHOVariants:
    """Standard vs. legacy operators, budget and initialisation."""

    @staticmethod
    def _weighted(binary):
        return float(np.dot(binary, np.linspace(-1.0, 1.0, binary.size)))

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"variant": "foo"}, "variant"),
            ({"init_inclusion": (0.0, 0.5)}, "init_inclusion"),
            ({"init_inclusion": (0.6, 0.5)}, "init_inclusion"),
            ({"mutation_rate": 1.5}, "mutation_rate"),
            ({"max_evaluations": 0}, "max_evaluations"),
            ({"position_bound": 0.0}, "position_bound"),
            ({"early_stop_patience": -1}, "early_stop_patience"),
        ],
    )
    def test_invalid_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            EHOOptimizer(**kwargs)

    def test_initial_subset_sizes_follow_inclusion_range(self):
        opt = EHOOptimizer(n_clans=20, clan_size=10, init_inclusion=(0.05, 0.5))
        positions = opt._initial_positions(np.random.default_rng(0), 200)
        share = (positions >= 0).mean(axis=2).ravel()
        assert share.min() < 0.12 and share.max() > 0.4
        assert share.max() < 0.62
        assert np.abs(positions).max() <= opt.position_bound

    def test_legacy_initialisation_is_half_of_pool(self):
        opt = EHOOptimizer(n_clans=20, clan_size=10, variant="legacy")
        share = (opt._initial_positions(np.random.default_rng(0), 200) >= 0).mean(axis=2)
        assert abs(share.mean() - 0.5) < 0.02 and share.std() < 0.06

    def test_standard_matriarch_is_beta_times_centre(self):
        opt = EHOOptimizer(n_clans=1, clan_size=4, beta=0.2, mutation_rate=0.0)
        clan = np.random.default_rng(1).standard_normal((4, 6))
        centre = clan.mean(axis=0)
        opt._update_clan(clan, step=0.5, rng=np.random.default_rng(2), mutation_rate=0.0)
        np.testing.assert_allclose(clan[0], 0.2 * centre)

    def test_mutation_flips_bits(self):
        opt = EHOOptimizer(n_clans=1, clan_size=3)
        clan = np.ones((3, 50))
        opt._update_clan(clan, step=0.0, rng=np.random.default_rng(0), mutation_rate=1.0)
        assert (clan[1:] < 0).all()

    def test_separating_uniform_within_bounds(self):
        opt = EHOOptimizer(n_clans=3, clan_size=2, position_bound=2.0)
        positions = np.zeros((3, 2, 1000))
        opt._separate(positions, np.random.default_rng(0))
        worst = positions[:, -1]
        assert np.abs(worst).max() <= 2.0 and worst.std() > 1.0
        assert (positions[:, 0] == 0).all()

    @pytest.mark.parametrize("variant", ["standard", "legacy"])
    def test_budget_is_respected(self, variant):
        calls = 0

        def objective(binary):
            nonlocal calls
            calls += 1
            return self._weighted(binary)

        opt = EHOOptimizer(n_clans=3, clan_size=4, max_iterations=1000,
                           early_stop_patience=0, variant=variant, max_evaluations=50)
        result = opt.optimize(objective, n_models=20, seed=0)
        assert calls == 50
        assert result.metadata["n_evaluations"] == 50
        assert result.metadata["budget_exhausted"] is True

    def test_best_subset_matches_best_fitness(self):
        opt = EHOOptimizer(n_clans=2, clan_size=5, max_iterations=10, early_stop_patience=0)
        result = opt.optimize(self._weighted, n_models=30, seed=3)
        assert result.best_fitness == pytest.approx(self._weighted(result.best_subset))
        assert result.best_fitness == max(result.convergence_history)

    def test_standard_beats_legacy_on_sparse_optimum(self):
        """The optimum selects 10 % of the models; standard EHO gets closer than legacy."""
        def objective(binary):
            return self._weighted(binary)

        scores = {}
        for variant in ("standard", "legacy"):
            opt = EHOOptimizer(n_clans=4, clan_size=5, max_iterations=30,
                               early_stop_patience=0, variant=variant)
            scores[variant] = np.mean([
                opt.optimize(objective, n_models=40, seed=s).best_fitness for s in range(5)
            ])
        assert scores["standard"] >= scores["legacy"]
