"""Optimizer protocol interface for ensemble subset selection.

Defines a Protocol-based interface that all optimizers must implement,
enabling seamless swapping of optimization algorithms (EHO, GA, PSO, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np


@dataclass
class OptimizationResult:
    """Result returned by any optimizer implementing OptimizerProtocol.

    Attributes
    ----------
    best_subset : np.ndarray
        Binary vector of length n_models (1=include, 0=exclude).
    best_fitness : float
        Fitness score of the best subset found.
    convergence_history : list[float]
        Best fitness at each iteration.
    n_iterations : int
        Total number of iterations executed.
    execution_time : float
        Wall-clock time in seconds.
    metadata : dict
        Optional algorithm-specific metadata (e.g., n_evaluations, clan info).
    """

    best_subset: np.ndarray
    best_fitness: float
    convergence_history: list[float]
    n_iterations: int
    execution_time: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_indices(self) -> list[int]:
        """Return indices of selected models."""
        return [int(i) for i in np.where(self.best_subset > 0.5)[0]]

    @property
    def n_selected(self) -> int:
        """Number of models selected in best subset."""
        return int(np.sum(self.best_subset > 0.5))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "best_subset": self.best_subset.tolist(),
            "best_indices": self.best_indices,
            "best_fitness": float(self.best_fitness),
            "n_selected": self.n_selected,
            "convergence_history": self.convergence_history,
            "n_iterations": self.n_iterations,
            "execution_time": round(self.execution_time, 4),
            "metadata": self.metadata,
        }


@runtime_checkable
class OptimizerProtocol(Protocol):
    """Protocol that all ensemble subset optimizers must implement.

    Any optimizer (EHO, GA, PSO, etc.) can be used as long as it
    implements the ``optimize`` method with the correct signature.

    Example
    -------
    >>> class MyOptimizer:
    ...     def optimize(self, objective_function, n_models, seed=42, **kwargs):
    ...         # ... custom optimization logic ...
    ...         return OptimizationResult(...)
    >>>
    >>> assert isinstance(MyOptimizer(), OptimizerProtocol)
    """

    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        n_models: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run optimization to find the best model subset.

        Parameters
        ----------
        objective_function : Callable[[np.ndarray], float]
            Takes a binary vector of length ``n_models`` (1=include, 0=exclude)
            and returns a fitness score. **Higher is better.**
        n_models : int
            Total number of models in the pool (length of binary vector).
        seed : int
            Random seed for reproducibility.
        **kwargs
            Additional algorithm-specific parameters.

        Returns
        -------
        OptimizationResult
            The optimization result with the best subset found.
        """
        ...


class BudgetExhausted(Exception):
    """Raised by :class:`EvaluationBudget` when the evaluation budget is used up."""


class EvaluationBudget:
    """Objective wrapper that counts evaluations and keeps the best solution.

    Every call counts as one evaluation (also when the objective answers from its
    cache), so optimizers compared at the same ``max_evaluations`` receive exactly
    the same number of fitness queries. A call beyond the budget raises
    :class:`BudgetExhausted`; optimizers catch it and return the best solution.

    Attributes
    ----------
    n_evaluations : int
    best_fitness : float
    best_subset : ndarray or None
        Binary vector of the best evaluated subset (first one on ties).
    trace : list[float]
        Best-so-far fitness after each evaluation (anytime curve).
    """

    def __init__(
        self,
        objective_function: Callable[[np.ndarray], float],
        max_evaluations: int | None = None,
        record_trace: bool = False,
    ) -> None:
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError(f"max_evaluations must be >= 1, got {max_evaluations}")
        self.objective_function = objective_function
        self.max_evaluations = max_evaluations
        self.record_trace = record_trace
        self.n_evaluations = 0
        self.best_fitness = -np.inf
        self.best_subset: np.ndarray | None = None
        self.trace: list[float] = []

    @property
    def exhausted(self) -> bool:
        return self.max_evaluations is not None and self.n_evaluations >= self.max_evaluations

    @property
    def remaining(self) -> int | None:
        return None if self.max_evaluations is None else self.max_evaluations - self.n_evaluations

    def __call__(self, binary: np.ndarray) -> float:
        if self.exhausted:
            raise BudgetExhausted
        fitness = float(self.objective_function(binary))
        self.n_evaluations += 1
        if fitness > self.best_fitness:
            self.best_fitness = fitness
            self.best_subset = np.asarray(binary, dtype=float).copy()
        if self.record_trace:
            self.trace.append(self.best_fitness)
        return fitness
