"""Tests for prunestack.optimization.protocol."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.optimization.protocol import OptimizationResult, OptimizerProtocol


class TestOptimizationResult:
    """Tests for the OptimizationResult dataclass."""

    def test_basic_creation(self):
        result = OptimizationResult(
            best_subset=np.array([1, 0, 1, 0, 1]),
            best_fitness=0.85,
            convergence_history=[0.5, 0.7, 0.85],
            n_iterations=3,
            execution_time=1.5,
        )
        assert result.best_fitness == 0.85
        assert result.n_iterations == 3
        assert result.execution_time == 1.5
        assert len(result.convergence_history) == 3

    def test_best_indices(self):
        result = OptimizationResult(
            best_subset=np.array([1, 0, 1, 0, 1]),
            best_fitness=0.85,
            convergence_history=[0.85],
            n_iterations=1,
            execution_time=0.1,
        )
        assert result.best_indices == [0, 2, 4]

    def test_n_selected(self):
        result = OptimizationResult(
            best_subset=np.array([1, 0, 1, 0, 1]),
            best_fitness=0.85,
            convergence_history=[0.85],
            n_iterations=1,
            execution_time=0.1,
        )
        assert result.n_selected == 3

    def test_to_dict(self):
        result = OptimizationResult(
            best_subset=np.array([1, 0, 1]),
            best_fitness=0.9,
            convergence_history=[0.5, 0.9],
            n_iterations=2,
            execution_time=0.5,
            metadata={"key": "value"},
        )
        d = result.to_dict()
        assert d["best_subset"] == [1, 0, 1]
        assert d["best_indices"] == [0, 2]
        assert d["best_fitness"] == 0.9
        assert d["n_selected"] == 2
        assert d["metadata"] == {"key": "value"}

    def test_empty_subset(self):
        result = OptimizationResult(
            best_subset=np.array([0, 0, 0]),
            best_fitness=0.0,
            convergence_history=[0.0],
            n_iterations=1,
            execution_time=0.1,
        )
        assert result.best_indices == []
        assert result.n_selected == 0

    def test_full_subset(self):
        result = OptimizationResult(
            best_subset=np.ones(10),
            best_fitness=0.7,
            convergence_history=[0.7],
            n_iterations=1,
            execution_time=0.1,
        )
        assert result.n_selected == 10
        assert result.best_indices == list(range(10))


class TestOptimizerProtocol:
    """Tests for the OptimizerProtocol type checking."""

    def test_eho_implements_protocol(self):
        from prunestack.optimization.eho import EHOOptimizer
        optimizer = EHOOptimizer()
        assert isinstance(optimizer, OptimizerProtocol)

    def test_custom_optimizer_implements_protocol(self):
        class CustomOptimizer:
            def optimize(self, objective_function, n_models, seed=42, **kwargs):
                binary = np.ones(n_models)
                return OptimizationResult(
                    best_subset=binary,
                    best_fitness=1.0,
                    convergence_history=[1.0],
                    n_iterations=1,
                    execution_time=0.0,
                )

        custom = CustomOptimizer()
        assert isinstance(custom, OptimizerProtocol)

    def test_non_optimizer_fails_protocol(self):
        class NotAnOptimizer:
            def do_something(self):
                pass

        assert not isinstance(NotAnOptimizer(), OptimizerProtocol)

    def test_missing_method_fails_protocol(self):
        class MissingOptimize:
            pass

        assert not isinstance(MissingOptimize(), OptimizerProtocol)
