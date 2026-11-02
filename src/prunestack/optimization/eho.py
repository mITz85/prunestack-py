"""Elephant Herding Optimization (EHO) for ensemble subset selection.

Binary adaptation of EHO (Wang, Deb & Coelho, 2015). Each elephant has a
continuous position x ∈ [−B, B]^M; model j is selected when x_j ≥ 0
(deterministic sigmoid threshold, sigmoid(x_j) ≥ 0.5).

Two variants are available:

``variant="standard"`` (default)
    * Size-diverse initialisation: elephant e draws an inclusion probability
      p_e ~ U(p_min, p_max) and x_ej ~ N(Φ⁻¹(p_e), 1), so P(model j selected) = p_e.
      Initial subsets therefore range from small to medium ensembles instead of
      all containing ≈ M/2 models.
    * Clan updating (Wang et al., 2015, Eq. 1): x ← x + α_t · (x_matriarch − x) · r,
      r ~ U(0, 1)^M, with the adaptive step α_t = α_EHO + (1 − α_EHO) · t/T.
    * Matriarch update (Wang et al., 2015, Eq. 2): x_matriarch ← β_EHO · x_centre.
    * Bit-flip mutation of the non-matriarch elephants: each bit is flipped
      (x_j ← −x_j) with probability ``mutation_rate`` (default 1/M).
    * Separating operator (Wang et al., 2015, Eq. 4): the worst elephant of each
      clan is re-sampled uniformly in [−B, B]^M.
    * Positions are clipped to [−B, B].

``variant="legacy"``
    The operator set of PruneStack ≤ 2.0.0 (standard-normal initialisation,
    matriarch moved by β_EHO·(x_centre − x)·r, worst elephant re-drawn from
    N(0, 1), no mutation, no bounds). Kept only to reproduce earlier results and
    for the optimizer benchmark.

In both variants the best subset ever evaluated is returned, and the search stops
after ``max_iterations``, after ``early_stop_patience`` iterations without
improvement, or when ``max_evaluations`` fitness evaluations are used.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
from scipy.stats import norm

from prunestack.optimization.protocol import BudgetExhausted, EvaluationBudget, OptimizationResult

logger = logging.getLogger(__name__)

VALID_EHO_VARIANTS = ("standard", "legacy")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _to_binary(
    continuous: np.ndarray,
    rng: np.random.Generator | None = None,
    stochastic: bool = False,
) -> np.ndarray:
    """Convert a continuous position to a binary selection vector.

    Deterministic threshold ``sigmoid(x) >= 0.5`` (i.e. ``x >= 0``) by default, so
    a position always maps to the same subset. ``stochastic=True`` samples each
    bit with probability ``sigmoid(x)`` instead.
    """
    probs = _sigmoid(continuous)
    if stochastic:
        if rng is None:
            rng = np.random.default_rng()
        return (rng.random(continuous.shape) < probs).astype(float)
    return (probs >= 0.5).astype(float)


def _ensure_valid(binary: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ensure at least one model is selected in the binary vector."""
    if np.sum(binary) < 1:
        idx = rng.integers(0, len(binary))
        binary[idx] = 1.0
    return binary


def _flip_bits(positions: np.ndarray, mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Flip the binary value of the masked coordinates (x_j -> -x_j, never 0)."""
    flipped = np.where(positions >= 0, -np.maximum(positions, eps), np.maximum(-positions, eps))
    return np.where(mask, flipped, positions)


class EHOOptimizer:
    """Elephant Herding Optimization for binary subset selection.

    Parameters
    ----------
    n_clans : int
        Number of clans (sub-populations).
    clan_size : int
        Number of elephants per clan.
    max_iterations : int
        Maximum number of generations T.
    alpha : float
        Matriarch influence α_EHO in [0, 1] (start value of the adaptive step α_t).
        Unrelated to the fitness weight α.
    beta : float
        Clan-centre factor β_EHO in [0, 1].
    early_stop_patience : int
        Stop after this many generations without improvement (0 = disabled).
    variant : 'standard' | 'legacy'
        Operator set, see the module docstring.
    init_inclusion : (p_min, p_max)
        Range of the per-elephant inclusion probability at initialisation
        (``standard`` only).
    mutation_rate : float or None
        Bit-flip probability per bit; ``None`` -> 1 / n_models (``standard`` only).
    max_evaluations : int or None
        Fitness-evaluation budget; ``None`` = unlimited.
    position_bound : float
        B, positions are kept in [−B, B] (``standard`` only).
    """

    def __init__(
        self,
        n_clans: int = 10,
        clan_size: int = 10,
        max_iterations: int = 100,
        alpha: float = 0.5,
        beta: float = 0.1,
        early_stop_patience: int = 25,
        variant: str = "standard",
        init_inclusion: tuple[float, float] = (0.05, 0.5),
        mutation_rate: float | None = None,
        max_evaluations: int | None = None,
        position_bound: float = 4.0,
    ) -> None:
        if n_clans < 1:
            raise ValueError(f"n_clans must be >= 1, got {n_clans}")
        if clan_size < 2:
            raise ValueError(f"clan_size must be >= 2, got {clan_size}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if early_stop_patience < 0:
            raise ValueError(f"early_stop_patience must be >= 0, got {early_stop_patience}")
        if variant not in VALID_EHO_VARIANTS:
            raise ValueError(f"variant must be one of {VALID_EHO_VARIANTS}, got {variant!r}")
        p_min, p_max = init_inclusion
        if not (0.0 < p_min <= p_max < 1.0):
            raise ValueError(f"init_inclusion must satisfy 0 < p_min <= p_max < 1, got {init_inclusion}")
        if mutation_rate is not None and not (0.0 <= mutation_rate <= 1.0):
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError(f"max_evaluations must be >= 1, got {max_evaluations}")
        if position_bound <= 0:
            raise ValueError(f"position_bound must be > 0, got {position_bound}")

        self.n_clans = n_clans
        self.clan_size = clan_size
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.early_stop_patience = early_stop_patience
        self.variant = variant
        self.init_inclusion = (float(p_min), float(p_max))
        self.mutation_rate = mutation_rate
        self.max_evaluations = max_evaluations
        self.position_bound = float(position_bound)

    @property
    def population_size(self) -> int:
        """Total number of elephants."""
        return self.n_clans * self.clan_size

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def _initial_positions(self, rng: np.random.Generator, n_models: int) -> np.ndarray:
        shape = (self.n_clans, self.clan_size, n_models)
        if self.variant == "legacy":
            return rng.standard_normal(shape)
        p = rng.uniform(*self.init_inclusion, size=(self.n_clans, self.clan_size, 1))
        positions = norm.ppf(p) + rng.standard_normal(shape)
        return np.clip(positions, -self.position_bound, self.position_bound)

    def _update_clan(
        self, clan: np.ndarray, step: float, rng: np.random.Generator, mutation_rate: float,
    ) -> None:
        """Clan updating, matriarch and (standard) mutation operators, in place."""
        n_models = clan.shape[1]
        matriarch = clan[0].copy()
        centre = clan.mean(axis=0)
        for e in range(1, self.clan_size):
            r = rng.random(n_models)
            clan[e] = clan[e] + step * (matriarch - clan[e]) * r
        if self.variant == "legacy":
            clan[0] = clan[0] + self.beta * (centre - clan[0]) * rng.random(n_models)
            return
        clan[0] = self.beta * centre
        mask = rng.random((self.clan_size - 1, n_models)) < mutation_rate
        clan[1:] = _flip_bits(clan[1:], mask)

    def _separate(self, positions: np.ndarray, rng: np.random.Generator) -> None:
        """Separating operator: replace the worst elephant of every clan, in place."""
        n_models = positions.shape[2]
        for c in range(self.n_clans):
            if self.variant == "legacy":
                positions[c, -1] = rng.standard_normal(n_models)
            else:
                positions[c, -1] = rng.uniform(-self.position_bound, self.position_bound, n_models)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        n_models: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run EHO and return the best subset ever evaluated.

        Parameters
        ----------
        objective_function : Callable[[np.ndarray], float]
            Takes binary vector (1=include, 0=exclude), returns fitness (higher is better).
        n_models : int
            Dimension of the binary vector (number of models).
        seed : int
            Random seed for reproducibility.
        """
        start_time = time.time()
        rng = np.random.default_rng(seed)
        budget = EvaluationBudget(objective_function, self.max_evaluations)
        mutation_rate = 1.0 / n_models if self.mutation_rate is None else self.mutation_rate

        logger.info(
            "EHO (%s) starting: %d clans x %d elephants, %d dimensions, max_iter=%d, max_evals=%s",
            self.variant, self.n_clans, self.clan_size, n_models, self.max_iterations,
            self.max_evaluations,
        )

        positions = self._initial_positions(rng, n_models)
        fitness = np.full((self.n_clans, self.clan_size), -np.inf)
        convergence_history: list[float] = []
        n_iterations = 0
        no_improve_count = 0
        early_stopped = False

        def evaluate_population() -> None:
            for c in range(self.n_clans):
                for e in range(self.clan_size):
                    binary = _ensure_valid(_to_binary(positions[c, e]), rng)
                    fitness[c, e] = budget(binary)

        try:
            evaluate_population()
            convergence_history.append(budget.best_fitness)
            for iteration in range(self.max_iterations):
                if self.max_evaluations is None:
                    progress = (iteration + 1) / self.max_iterations
                else:
                    progress = min(1.0, budget.n_evaluations / self.max_evaluations)
                step = self.alpha + (1.0 - self.alpha) * progress  # adaptive alpha_t

                for c in range(self.n_clans):  # best elephant (matriarch) first
                    order = np.argsort(-fitness[c])
                    positions[c] = positions[c][order]
                    fitness[c] = fitness[c][order]
                for c in range(self.n_clans):
                    self._update_clan(positions[c], step, rng, mutation_rate)
                self._separate(positions, rng)
                if self.variant == "standard":
                    np.clip(positions, -self.position_bound, self.position_bound, out=positions)

                best_before = budget.best_fitness
                n_iterations = iteration + 1
                evaluate_population()
                convergence_history.append(budget.best_fitness)

                no_improve_count = 0 if budget.best_fitness > best_before else no_improve_count + 1
                if 0 < self.early_stop_patience <= no_improve_count:
                    early_stopped = True
                    logger.info("EHO early stop at iteration %d (best=%.6f)", n_iterations, budget.best_fitness)
                    break
        except BudgetExhausted:
            convergence_history.append(budget.best_fitness)
            logger.info("EHO evaluation budget of %d used at iteration %d", budget.n_evaluations, n_iterations)

        execution_time = time.time() - start_time
        best_subset = budget.best_subset
        logger.info(
            "EHO completed: fitness=%.6f, selected=%d/%d models, %d iterations, %.2fs, %d evaluations",
            budget.best_fitness, int(best_subset.sum()), n_models, n_iterations,
            execution_time, budget.n_evaluations,
        )
        return OptimizationResult(
            best_subset=best_subset,
            best_fitness=float(budget.best_fitness),
            convergence_history=[float(v) for v in convergence_history],
            n_iterations=n_iterations,
            execution_time=execution_time,
            metadata={
                "n_evaluations": budget.n_evaluations,
                "variant": self.variant,
                "n_clans": self.n_clans,
                "clan_size": self.clan_size,
                "alpha": self.alpha,
                "beta": self.beta,
                "mutation_rate": mutation_rate if self.variant == "standard" else 0.0,
                "init_inclusion": list(self.init_inclusion) if self.variant == "standard" else None,
                "max_evaluations": self.max_evaluations,
                "early_stopped": early_stopped,
                "budget_exhausted": budget.exhausted,
            },
        )

    def __repr__(self) -> str:
        return (
            f"EHOOptimizer(variant='{self.variant}', n_clans={self.n_clans}, "
            f"clan_size={self.clan_size}, max_iterations={self.max_iterations}, "
            f"alpha={self.alpha}, beta={self.beta}, max_evaluations={self.max_evaluations})"
        )
