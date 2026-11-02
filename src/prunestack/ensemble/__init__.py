"""prunestack.ensemble - stacking, subset evaluation and strategy comparison."""

from prunestack.ensemble.evaluator import EnsembleEvaluator, evaluate_baseline
from prunestack.ensemble.stacking import StackedEnsemble, available_meta_learners
from prunestack.ensemble.strategies import compare_ensemble_strategies

__all__ = [
    "EnsembleEvaluator",
    "StackedEnsemble",
    "available_meta_learners",
    "compare_ensemble_strategies",
    "evaluate_baseline",
]
