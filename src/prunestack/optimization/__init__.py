"""prunestack.optimization — optimizer interface and implementations.

Provides the OptimizerProtocol for pluggable optimizers, the default
EHO (Elephant Herding Optimization) implementation, the evaluation budget
and the baseline optimizers used in the equal-budget benchmark.
"""

from prunestack.optimization.baselines import (
    BinaryPSO,
    GeneticAlgorithm,
    GreedyForwardSelection,
    HillClimbing,
    RandomSearch,
    SimulatedAnnealing,
)
from prunestack.optimization.eho import EHOOptimizer
from prunestack.optimization.objective import FitnessObjective
from prunestack.optimization.protocol import (
    BudgetExhausted,
    EvaluationBudget,
    OptimizationResult,
    OptimizerProtocol,
)
from prunestack.optimization.runner import optimize_alpha_grid, optimize_ensemble

__all__ = [
    "OptimizationResult",
    "OptimizerProtocol",
    "EvaluationBudget",
    "BudgetExhausted",
    "EHOOptimizer",
    "RandomSearch",
    "GreedyForwardSelection",
    "SimulatedAnnealing",
    "HillClimbing",
    "GeneticAlgorithm",
    "BinaryPSO",
    "FitnessObjective",
    "optimize_ensemble",
    "optimize_alpha_grid",
]
