"""Baseline optimizers for the equal-budget optimizer benchmark.

All optimizers implement :class:`~prunestack.optimization.protocol.OptimizerProtocol`
and share the same conventions as :class:`~prunestack.optimization.eho.EHOOptimizer`:

* a solution is a binary vector over the model pool (empty subsets are repaired
  to one random model);
* every objective call counts against ``max_evaluations`` through
  :class:`~prunestack.optimization.protocol.EvaluationBudget`, and the best subset
  ever evaluated is returned;
* stochastic optimizers start from size-diverse random subsets: each solution
  draws its own inclusion probability p ~ U(init_inclusion) (the same scheme as
  the standard EHO initialisation);
* an *iteration* is one generation / swarm step (population methods), one block
  of ``proposals_per_iteration`` proposals (local search and random search) or
  one forward step (greedy); ``convergence_history`` holds the best fitness after
  each iteration.

Implemented baselines
---------------------
RandomSearch
    Independent size-diverse random subsets.
GreedyForwardSelection
    Deterministic ordered pruning: start from the empty ensemble and add the
    model that most improves the fitness until no addition improves it.
SimulatedAnnealing / HillClimbing
    Single-bit-flip local search; geometric cooling of the temperature
    (Kirkpatrick et al., 1983). ``HillClimbing`` accepts improvements only.
GeneticAlgorithm
    Generational GA with tournament selection, uniform crossover, bit-flip
    mutation (1/M) and one elite.
BinaryPSO
    Binary particle swarm optimisation (Kennedy & Eberhart, 1997) with the
    constriction-equivalent coefficients of Clerc & Kennedy (2002) and velocity
    clamping.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any, Callable

import numpy as np

from prunestack.metrics.fitness import sample_random_subsets
from prunestack.optimization.eho import _ensure_valid, _sigmoid
from prunestack.optimization.protocol import BudgetExhausted, EvaluationBudget, OptimizationResult

logger = logging.getLogger(__name__)


class _BudgetedOptimizer:
    """Shared driver: budget, iteration limit, history and result construction.

    Subclasses implement :meth:`_iterations`, a generator that performs one
    iteration per ``yield`` and evaluates solutions only through ``budget``.
    """

    name = "base"

    def __init__(
        self,
        max_evaluations: int | None = None,
        max_iterations: int = 100,
        init_inclusion: tuple[float, float] = (0.05, 0.5),
    ) -> None:
        if max_evaluations is not None and max_evaluations < 1:
            raise ValueError(f"max_evaluations must be >= 1, got {max_evaluations}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        p_min, p_max = init_inclusion
        if not (0.0 < p_min <= p_max < 1.0):
            raise ValueError(f"init_inclusion must satisfy 0 < p_min <= p_max < 1, got {init_inclusion}")
        self.max_evaluations = max_evaluations
        self.max_iterations = max_iterations
        self.init_inclusion = (float(p_min), float(p_max))

    def _random_subsets(self, rng: np.random.Generator, n: int, n_models: int) -> np.ndarray:
        return sample_random_subsets(rng, n, n_models, self.init_inclusion).astype(float)

    def _progress(self, budget: EvaluationBudget, iteration: int) -> float:
        """Fraction of the search used, by evaluations if budgeted, else by iterations."""
        if self.max_evaluations is not None:
            return min(1.0, budget.n_evaluations / self.max_evaluations)
        return min(1.0, iteration / self.max_iterations)

    def _iterations(
        self, budget: EvaluationBudget, n_models: int, rng: np.random.Generator,
    ) -> Iterator[None]:
        raise NotImplementedError

    def _metadata(self) -> dict[str, Any]:
        return {}

    def optimize(
        self,
        objective_function: Callable[[np.ndarray], float],
        n_models: int,
        seed: int = 42,
        **kwargs: Any,
    ) -> OptimizationResult:
        """Run the search and return the best subset ever evaluated."""
        start_time = time.time()
        rng = np.random.default_rng(seed)
        budget = EvaluationBudget(objective_function, self.max_evaluations)
        history: list[float] = []
        n_iterations = 0
        try:
            for _ in self._iterations(budget, n_models, rng):
                n_iterations += 1
                history.append(budget.best_fitness)
                if n_iterations >= self.max_iterations:
                    break
        except BudgetExhausted:
            history.append(budget.best_fitness)

        execution_time = time.time() - start_time
        logger.info(
            "%s completed: fitness=%.6f, selected=%d/%d, %d iterations, %d evaluations, %.2fs",
            self.name, budget.best_fitness, int(budget.best_subset.sum()), n_models,
            n_iterations, budget.n_evaluations, execution_time,
        )
        return OptimizationResult(
            best_subset=budget.best_subset,
            best_fitness=float(budget.best_fitness),
            convergence_history=[float(v) for v in history],
            n_iterations=n_iterations,
            execution_time=execution_time,
            metadata={
                "optimizer": self.name,
                "n_evaluations": budget.n_evaluations,
                "max_evaluations": self.max_evaluations,
                "budget_exhausted": budget.exhausted,
                **self._metadata(),
            },
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(max_evaluations={self.max_evaluations}, max_iterations={self.max_iterations})"


class RandomSearch(_BudgetedOptimizer):
    """Independent size-diverse random subsets, ``proposals_per_iteration`` per iteration."""

    name = "random_search"

    def __init__(self, proposals_per_iteration: int = 100, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if proposals_per_iteration < 1:
            raise ValueError(f"proposals_per_iteration must be >= 1, got {proposals_per_iteration}")
        self.proposals_per_iteration = proposals_per_iteration

    def _iterations(self, budget, n_models, rng):
        while True:
            for binary in self._random_subsets(rng, self.proposals_per_iteration, n_models):
                budget(binary)
            yield


class GreedyForwardSelection(_BudgetedOptimizer):
    """Ordered (forward) pruning: add the best model until no addition helps.

    Deterministic; ties are broken by the lowest model index. Stops at the
    first step without strict improvement, so it may use fewer evaluations
    than the budget (at most M(M+1)/2).
    """

    name = "greedy_forward"

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("max_iterations", 10_000)
        super().__init__(**kwargs)

    def _iterations(self, budget, n_models, rng):
        selected = np.zeros(n_models)
        current = -np.inf
        while selected.sum() < n_models:
            candidates = np.flatnonzero(selected == 0)
            scores = np.empty(len(candidates))
            for k, j in enumerate(candidates):
                trial = selected.copy()
                trial[j] = 1.0
                scores[k] = budget(trial)
            best = int(np.argmax(scores))
            if scores[best] <= current:
                return
            selected[candidates[best]] = 1.0
            current = scores[best]
            yield


class SimulatedAnnealing(_BudgetedOptimizer):
    """Single-bit-flip simulated annealing.

    A proposal flips one uniformly chosen bit (proposals that would empty the
    subset are skipped). A worse proposal is accepted with probability
    exp(Δ / T); the temperature decays geometrically from ``initial_temperature``
    to ``final_temperature`` over the budget (or over ``max_iterations``).
    ``initial_temperature = 0`` gives first-improvement hill climbing.
    The defaults are set for fitness values normalised to roughly [0, 1].
    """

    name = "simulated_annealing"

    def __init__(
        self,
        initial_temperature: float = 0.01,
        final_temperature: float = 1e-5,
        proposals_per_iteration: int = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if initial_temperature < 0 or final_temperature < 0:
            raise ValueError("temperatures must be >= 0")
        if initial_temperature > 0 and not (0 < final_temperature <= initial_temperature):
            raise ValueError("final_temperature must be in (0, initial_temperature]")
        if proposals_per_iteration < 1:
            raise ValueError(f"proposals_per_iteration must be >= 1, got {proposals_per_iteration}")
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.proposals_per_iteration = proposals_per_iteration

    def _temperature(self, progress: float) -> float:
        if self.initial_temperature == 0:
            return 0.0
        return self.initial_temperature * (self.final_temperature / self.initial_temperature) ** progress

    def _iterations(self, budget, n_models, rng):
        current = self._random_subsets(rng, 1, n_models)[0]
        current_fitness = budget(current)
        iteration = 0
        while True:
            for _ in range(self.proposals_per_iteration):
                j = rng.integers(n_models)
                if current[j] == 1.0 and current.sum() == 1:
                    continue
                trial = current.copy()
                trial[j] = 1.0 - trial[j]
                trial_fitness = budget(trial)
                delta = trial_fitness - current_fitness
                temperature = self._temperature(self._progress(budget, iteration))
                if delta > 0 or (temperature > 0 and rng.random() < np.exp(delta / temperature)):
                    current, current_fitness = trial, trial_fitness
            iteration += 1
            yield

    def _metadata(self):
        return {"initial_temperature": self.initial_temperature,
                "final_temperature": self.final_temperature}


class HillClimbing(SimulatedAnnealing):
    """First-improvement single-bit-flip hill climbing (simulated annealing at T = 0)."""

    name = "hill_climbing"

    def __init__(self, proposals_per_iteration: int = 100, **kwargs: Any) -> None:
        super().__init__(initial_temperature=0.0, final_temperature=0.0,
                         proposals_per_iteration=proposals_per_iteration, **kwargs)


class GeneticAlgorithm(_BudgetedOptimizer):
    """Generational GA: tournament selection, uniform crossover, bit-flip mutation, elitism."""

    name = "genetic_algorithm"

    def __init__(
        self,
        population_size: int = 100,
        tournament_size: int = 3,
        crossover_rate: float = 0.9,
        mutation_rate: float | None = None,
        n_elites: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if population_size < 2:
            raise ValueError(f"population_size must be >= 2, got {population_size}")
        if tournament_size < 1:
            raise ValueError(f"tournament_size must be >= 1, got {tournament_size}")
        if not 0.0 <= crossover_rate <= 1.0:
            raise ValueError(f"crossover_rate must be in [0, 1], got {crossover_rate}")
        if mutation_rate is not None and not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must be in [0, 1], got {mutation_rate}")
        if not 0 <= n_elites < population_size:
            raise ValueError(f"n_elites must be in [0, population_size), got {n_elites}")
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_elites = n_elites

    def _tournament(self, fitness: np.ndarray, rng: np.random.Generator) -> int:
        contestants = rng.integers(len(fitness), size=self.tournament_size)
        return int(contestants[np.argmax(fitness[contestants])])

    def _iterations(self, budget, n_models, rng):
        mutation_rate = 1.0 / n_models if self.mutation_rate is None else self.mutation_rate
        population = self._random_subsets(rng, self.population_size, n_models)
        fitness = np.array([budget(ind) for ind in population])
        while True:
            elites = np.argsort(-fitness)[: self.n_elites]
            children = np.empty((self.population_size - self.n_elites, n_models))
            for k in range(len(children)):
                a = population[self._tournament(fitness, rng)]
                b = population[self._tournament(fitness, rng)]
                if rng.random() < self.crossover_rate:
                    a = np.where(rng.random(n_models) < 0.5, a, b)
                flip = rng.random(n_models) < mutation_rate
                children[k] = _ensure_valid(np.where(flip, 1.0 - a, a), rng)
            child_fitness = np.array([budget(c) for c in children])
            population = np.vstack([population[elites], children])
            fitness = np.concatenate([fitness[elites], child_fitness])
            yield

    def _metadata(self):
        return {"population_size": self.population_size, "tournament_size": self.tournament_size,
                "crossover_rate": self.crossover_rate, "mutation_rate": self.mutation_rate,
                "n_elites": self.n_elites}


class BinaryPSO(_BudgetedOptimizer):
    """Binary PSO (Kennedy & Eberhart, 1997).

    v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x), clipped to [−v_max, v_max];
    x_j = 1 with probability sigmoid(v_j). Initial velocities equal logit(p_i)
    for a particle-specific p_i ~ U(init_inclusion), so the first moves keep the
    size diversity of the initial swarm instead of collapsing to p = 0.5.
    """

    name = "binary_pso"

    def __init__(
        self,
        swarm_size: int = 100,
        inertia: float = 0.7298,
        cognitive: float = 1.49618,
        social: float = 1.49618,
        max_velocity: float = 4.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if swarm_size < 2:
            raise ValueError(f"swarm_size must be >= 2, got {swarm_size}")
        if max_velocity <= 0:
            raise ValueError(f"max_velocity must be > 0, got {max_velocity}")
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_velocity = max_velocity

    def _iterations(self, budget, n_models, rng):
        p = rng.uniform(*self.init_inclusion, size=(self.swarm_size, 1))
        velocity = np.repeat(np.log(p / (1.0 - p)), n_models, axis=1)
        x = (rng.random((self.swarm_size, n_models)) < _sigmoid(velocity)).astype(float)
        for row in x:
            _ensure_valid(row, rng)
        fitness = np.array([budget(row) for row in x])
        pbest, pbest_fitness = x.copy(), fitness.copy()
        while True:
            gbest = pbest[np.argmax(pbest_fitness)]
            r1 = rng.random(x.shape)
            r2 = rng.random(x.shape)
            velocity = self.inertia * velocity + self.cognitive * r1 * (pbest - x) + self.social * r2 * (gbest - x)
            np.clip(velocity, -self.max_velocity, self.max_velocity, out=velocity)
            x = (rng.random(x.shape) < _sigmoid(velocity)).astype(float)
            for i, row in enumerate(x):
                fitness[i] = budget(_ensure_valid(row, rng))
            improved = fitness > pbest_fitness
            pbest[improved] = x[improved]
            pbest_fitness[improved] = fitness[improved]
            yield

    def _metadata(self):
        return {"swarm_size": self.swarm_size, "inertia": self.inertia, "cognitive": self.cognitive,
                "social": self.social, "max_velocity": self.max_velocity}
