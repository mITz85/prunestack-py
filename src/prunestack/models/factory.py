"""Construction of the heterogeneous base-model pool."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from prunestack.models.feature_subsets import generate_feature_subsets
from prunestack.models.registry import get_model_class

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for a single base model in the pool.

    Attributes
    ----------
    model_id : int
        Unique integer identifier within the pool.
    name : str
        Human-readable name (e.g., 'DecisionTreeClassifier_original').
    algorithm : str
        Algorithm name (registry key).
    params : dict[str, Any]
        Hyperparameters for model instantiation.
    feature_indices : list[int]
        Indices of features this model uses.
    n_features_used : int
        Number of features used by this model.
    n_features_total : int
        Total features in the dataset.
    is_original : bool
        True if this model uses all features.
    seed : int
        Seed of the feature-subset draw (``base_seed`` for original models).
    """

    model_id: int
    name: str
    algorithm: str
    params: dict[str, Any]
    feature_indices: list[int]
    n_features_used: int
    n_features_total: int
    is_original: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "algorithm": self.algorithm,
            "params": {k: _serialize_value(v) for k, v in self.params.items()},
            "feature_indices": self.feature_indices,
            "n_features_used": self.n_features_used,
            "n_features_total": self.n_features_total,
            "is_original": self.is_original,
            "seed": self.seed,
        }


def _serialize_value(v: Any) -> Any:
    """Make a value JSON-serializable."""
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def create_model_pool(
    n_features: int,
    hyperparameters: dict[str, dict[str, Any]],
    task_type: str,
    base_seed: int = 42,
    max_models_per_algo: int = 20,
    feature_subset_min_ratio: float = 0.6,
    feature_subset_max_ratio: float = 0.9,
) -> list[ModelConfig]:
    """Create the full base model pool.

    For each algorithm in ``hyperparameters`` (see
    :func:`prunestack.models.hyperparameters.load_hyperparameters`) the pool holds
    one original model trained on all features and ``max_models_per_algo - 1``
    feature-subset variants. All models of an algorithm share the same fixed
    hyperparameters; they differ only in the features they see.

    Parameters
    ----------
    n_features : int
        Total number of features in the dataset.
    hyperparameters : dict[str, dict[str, Any]]
        Algorithm name -> fixed hyperparameters.
    task_type : str
        'classification' or 'regression'.
    base_seed : int
        Global random seed.
    max_models_per_algo : int
        Total models per algorithm (1 original + rest subsets).
    feature_subset_min_ratio : float
        Min fraction of features for subsets.
    feature_subset_max_ratio : float
        Max fraction of features for subsets.

    Returns
    -------
    list[ModelConfig]
        The full model pool.
    """
    algorithms = sorted(hyperparameters)
    n_variants = max_models_per_algo - 1
    model_pool: list[ModelConfig] = []
    model_id = 0

    for algo_idx, algo_name in enumerate(algorithms):
        params = hyperparameters[algo_name]

        # Validate algorithm is in registry
        get_model_class(algo_name, task_type)

        # Original model (all features)
        original = ModelConfig(
            model_id=model_id,
            name=f"{algo_name}_original",
            algorithm=algo_name,
            params=dict(params),
            feature_indices=list(range(n_features)),
            n_features_used=n_features,
            n_features_total=n_features,
            is_original=True,
            seed=base_seed,
        )
        model_pool.append(original)
        model_id += 1

        # Feature subset variants
        subsets = generate_feature_subsets(
            n_features=n_features,
            n_variants=n_variants,
            base_seed=base_seed,
            algo_idx=algo_idx,
            min_ratio=feature_subset_min_ratio,
            max_ratio=feature_subset_max_ratio,
        )

        for v_idx, indices in enumerate(subsets):
            variant = ModelConfig(
                model_id=model_id,
                name=f"{algo_name}_subset_{v_idx + 1:02d}",
                algorithm=algo_name,
                params=dict(params),
                feature_indices=indices,
                n_features_used=len(indices),
                n_features_total=n_features,
                is_original=False,
                seed=base_seed + algo_idx * 1000 + v_idx,
            )
            model_pool.append(variant)
            model_id += 1

        logger.info(
            "  %s: 1 original + %d subsets = %d models",
            algo_name, len(subsets), 1 + len(subsets),
        )

    logger.info("Total base models in pool: %d", len(model_pool))
    return model_pool
