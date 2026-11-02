"""Tests for the evaluation budget and the baseline optimizers."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.optimization import (
    BinaryPSO,
    BudgetExhausted,
    EHOOptimizer,
    EvaluationBudget,
    GeneticAlgorithm,
    GreedyForwardSelection,
    HillClimbing,
    OptimizerProtocol,
    RandomSearch,
    SimulatedAnnealing,
)

N_MODELS = 20
TARGET = np.zeros(N_MODELS)
TARGET[[1, 4, 9]] = 1.0


def sparse_objective(binary: np.ndarray) -> float:
    """Maximum 0 at the 3-model subset TARGET; empty subsets are infeasible."""
    if binary.sum() == 0:
        return -np.inf
    return -float(np.sum(np.abs(binary - TARGET)))


STOCHASTIC = [RandomSearch, HillClimbing, SimulatedAnnealing, GeneticAlgorithm, BinaryPSO]


class TestEvaluationBudget:
    def test_counts_tracks_best_and_trace(self):
        budget = EvaluationBudget(lambda b: float(b.sum()), max_evaluations=3, record_trace=True)
        assert budget(np.array([1.0, 0.0])) == 1.0
        budget(np.array([1.0, 1.0]))
        budget(np.array([0.0, 1.0]))
        assert budget.n_evaluations == 3 and budget.exhausted and budget.remaining == 0
        assert budget.best_fitness == 2.0
        np.testing.assert_array_equal(budget.best_subset, [1.0, 1.0])
        assert budget.trace == [1.0, 2.0, 2.0]
        with pytest.raises(BudgetExhausted):
            budget(np.array([1.0, 1.0]))
        assert budget.n_evaluations == 3

    def test_first_best_kept_on_ties_and_unlimited(self):
        budget = EvaluationBudget(lambda b: 1.0)
        budget(np.array([1.0, 0.0]))
        budget(np.array([0.0, 1.0]))
        np.testing.assert_array_equal(budget.best_subset, [1.0, 0.0])
        assert budget.remaining is None and not budget.exhausted

    def test_invalid_budget(self):
        with pytest.raises(ValueError):
            EvaluationBudget(lambda b: 0.0, max_evaluations=0)


class TestBaselines:
    @pytest.mark.parametrize("cls", [*STOCHASTIC, GreedyForwardSelection])
    def test_protocol_budget_and_result(self, cls):
        opt = cls(max_evaluations=300, max_iterations=10**6)
        assert isinstance(opt, OptimizerProtocol)
        calls = []
        result = opt.optimize(lambda b: calls.append(1) or sparse_objective(b), N_MODELS, seed=1)
        assert len(calls) == result.metadata["n_evaluations"] <= 300
        assert result.best_fitness == sparse_objective(result.best_subset)
        assert result.n_selected >= 1
        assert result.convergence_history == sorted(result.convergence_history)

    @pytest.mark.parametrize("cls", STOCHASTIC)
    def test_reproducible(self, cls):
        r1 = cls(max_evaluations=200).optimize(sparse_objective, N_MODELS, seed=5)
        r2 = cls(max_evaluations=200).optimize(sparse_objective, N_MODELS, seed=5)
        np.testing.assert_array_equal(r1.best_subset, r2.best_subset)
        assert r1.convergence_history == r2.convergence_history

    @pytest.mark.parametrize("cls", [HillClimbing, SimulatedAnnealing, GeneticAlgorithm, GreedyForwardSelection])
    def test_finds_sparse_optimum(self, cls):
        result = cls(max_evaluations=3000, max_iterations=10**6).optimize(sparse_objective, N_MODELS, seed=0)
        np.testing.assert_array_equal(result.best_subset, TARGET)

    def test_greedy_stops_without_improvement(self):
        result = GreedyForwardSelection().optimize(sparse_objective, N_MODELS)
        assert result.n_iterations == 3
        # 20 + 19 + 18 additions, then 17 non-improving candidates
        assert result.metadata["n_evaluations"] == 20 + 19 + 18 + 17
        assert not result.metadata["budget_exhausted"]

    def test_iteration_limit(self):
        result = RandomSearch(proposals_per_iteration=10, max_iterations=4).optimize(sparse_objective, N_MODELS)
        assert result.n_iterations == 4 and result.metadata["n_evaluations"] == 40

    def test_random_search_respects_init_inclusion(self):
        sizes = []
        RandomSearch(max_evaluations=500, init_inclusion=(0.1, 0.2)).optimize(
            lambda b: sizes.append(b.sum()) or 0.0, 100, seed=0)
        assert 5 <= np.mean(sizes) <= 25

    @pytest.mark.parametrize("kwargs", [
        {"max_evaluations": 0}, {"max_iterations": 0}, {"init_inclusion": (0.5, 0.1)},
    ])
    def test_invalid_common_params(self, kwargs):
        with pytest.raises(ValueError):
            RandomSearch(**kwargs)

    def test_invalid_specific_params(self):
        with pytest.raises(ValueError):
            GeneticAlgorithm(population_size=1)
        with pytest.raises(ValueError):
            GeneticAlgorithm(n_elites=100, population_size=100)
        with pytest.raises(ValueError):
            BinaryPSO(max_velocity=0)
        with pytest.raises(ValueError):
            SimulatedAnnealing(initial_temperature=0.01, final_temperature=0.1)

    def test_equal_budget_with_eho(self):
        for opt in (EHOOptimizer(max_evaluations=250, max_iterations=10**6, early_stop_patience=0),
                    GeneticAlgorithm(max_evaluations=250, max_iterations=10**6)):
            tracker = EvaluationBudget(sparse_objective, record_trace=True)
            opt.optimize(tracker, N_MODELS, seed=3)
            assert tracker.n_evaluations == 250 and len(tracker.trace) == 250
