"""Building blocks shared by the nested and the in-sample experiment protocols."""

from __future__ import annotations

from typing import Any

from prunestack.config import ExperimentConfig
from prunestack.models.hyperparameters import load_stacking_config
from prunestack.optimization.eho import EHOOptimizer


def build_optimizer(cfg: ExperimentConfig) -> EHOOptimizer:
    """EHO optimizer with the settings of ``cfg``."""
    return EHOOptimizer(
        n_clans=cfg.eho_n_clans,
        clan_size=cfg.eho_clan_size,
        max_iterations=cfg.eho_max_iterations,
        alpha=cfg.eho_alpha,
        beta=cfg.eho_beta,
        early_stop_patience=cfg.eho_early_stop_patience,
        variant=cfg.eho_variant,
        init_inclusion=(cfg.eho_init_inclusion_min, cfg.eho_init_inclusion_max),
        mutation_rate=cfg.eho_mutation_rate,
        max_evaluations=cfg.eho_max_evaluations,
    )


def stacking_settings(cfg: ExperimentConfig, task_type: str) -> tuple[str, dict[str, Any]]:
    """Meta-learner name and parameters (CLI/YAML ``stacking_method`` overrides the JSON)."""
    meta_learner, params = load_stacking_config(task_type, cfg.hyperparameters_path)
    if cfg.stacking_method is not None and cfg.stacking_method != meta_learner:
        meta_learner, params = cfg.stacking_method, {}
    return meta_learner, params
