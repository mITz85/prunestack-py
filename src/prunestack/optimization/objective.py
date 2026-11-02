"""Objective function wrapper for ensemble subset optimization.

Wraps :func:`prunestack.metrics.fitness.calculate_fitness` into a callable
that maps a binary selection vector to a fitness score, with caching so that
identical subsets are evaluated only once.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from prunestack.metrics.fitness import FitnessScaler, calculate_fitness

logger = logging.getLogger(__name__)


class FitnessObjective:
    """Callable objective function for optimizer consumption.

    Takes a binary vector and returns
    ``alpha * P~(S) + (1 - alpha) * D~(S)`` for the selected subset ``S``
    (P and D rescaled by ``scaler``). Results are cached by the binary vector.
    The empty subset is infeasible and scores ``-inf`` (``+inf`` when minimizing).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels / target values.
    predictions_matrix : np.ndarray
        Out-of-fold predictions, shape (n_samples, n_models).
    alpha : float
        Performance weight; diversity is weighted by ``1 - alpha``.
    task_type : str
        'classification' or 'regression'.
    maximize : bool
        If True (default), return fitness as-is (higher is better).
        If False, return negative fitness (for minimization frameworks).
    performance_metric : str or None
        Named performance metric (task default if None).
    diversity_metric : str
        Diversity measure (``'auto'``: disagreement for classification,
        ambiguity for regression).
    scaler : FitnessScaler or None
        Rescaling of P and D (see :func:`prunestack.metrics.fitness.fit_fitness_scaler`);
        ``None`` = raw P and D.
    """

    def __init__(
        self,
        y_true: np.ndarray,
        predictions_matrix: np.ndarray,
        alpha: float,
        task_type: str,
        maximize: bool = True,
        performance_metric: str | None = None,
        diversity_metric: str = "auto",
        scaler: FitnessScaler | None = None,
    ) -> None:
        self.y_true = np.asarray(y_true)
        self.predictions_matrix = np.asarray(predictions_matrix, dtype=float)
        self.alpha = alpha
        self.task_type = task_type
        self.maximize = maximize
        self.performance_metric = performance_metric
        self.diversity_metric = diversity_metric
        self.scaler = scaler
        self.n_models = self.predictions_matrix.shape[1]

        # Cache: tuple(binary_vector) -> fitness
        self._cache: dict[tuple[int, ...], float] = {}
        self._n_calls = 0
        self._n_cache_hits = 0

    def __call__(self, binary_subset: np.ndarray) -> float:
        """Evaluate the fitness of a binary subset.

        Parameters
        ----------
        binary_subset : np.ndarray
            Binary vector of length n_models (1=include, 0=exclude).

        Returns
        -------
        float
            Fitness score. Higher is better if maximize=True.
        """
        self._n_calls += 1

        # Round to clean binary
        binary = (np.asarray(binary_subset) > 0.5).astype(int)

        # Cache lookup
        key = tuple(binary.tolist())
        if key in self._cache:
            self._n_cache_hits += 1
            return self._cache[key]

        # Convert to indices
        indices = [int(i) for i in np.where(binary == 1)[0]]

        if len(indices) == 0:
            fitness = -np.inf if self.maximize else np.inf
            self._cache[key] = fitness
            return fitness

        raw_fitness = calculate_fitness(
            self.y_true,
            self.predictions_matrix,
            indices,
            self.alpha,
            self.task_type,
            performance_metric=self.performance_metric,
            diversity_metric=self.diversity_metric,
            scaler=self.scaler,
        )

        fitness = raw_fitness if self.maximize else -raw_fitness
        self._cache[key] = fitness
        return fitness

    @property
    def n_calls(self) -> int:
        """Total number of calls to the objective."""
        return self._n_calls

    @property
    def n_cache_hits(self) -> int:
        """Number of cache hits."""
        return self._n_cache_hits

    @property
    def cache_size(self) -> int:
        """Number of unique subsets evaluated."""
        return len(self._cache)

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of calls served from cache."""
        if self._n_calls == 0:
            return 0.0
        return self._n_cache_hits / self._n_calls

    def get_best_cached(self) -> tuple[tuple[int, ...], float] | None:
        """Return the best (subset_key, fitness) from cache, or None."""
        if not self._cache:
            return None
        if self.maximize:
            best_key = max(self._cache, key=self._cache.get)  # type: ignore
        else:
            best_key = min(self._cache, key=self._cache.get)  # type: ignore
        return best_key, self._cache[best_key]

    def reset_cache(self) -> None:
        """Clear the evaluation cache and counters."""
        self._cache.clear()
        self._n_calls = 0
        self._n_cache_hits = 0

    def __repr__(self) -> str:
        return (
            f"FitnessObjective(alpha={self.alpha}, "
            f"task_type='{self.task_type}', n_models={self.n_models}, "
            f"calls={self._n_calls}, cache_size={self.cache_size})"
        )
