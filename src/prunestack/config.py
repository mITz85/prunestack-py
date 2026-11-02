"""Experiment configuration: defaults, YAML loading and validation.

Priority when building a configuration: command-line arguments > YAML file > ``DEFAULTS``.
The defaults reproduce the protocol reported in the accompanying manuscript.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_ALPHA_GRID: list[float] = [round(i * 0.1, 1) for i in range(11)]

DEFAULTS: dict[str, Any] = {
    "seed": 1985,
    "n_folds": 5,
    "task_type": "auto",
    # Evaluation protocol. "nested" (manuscript): an outer k-fold split separates the rows
    # on which the subset is selected (inner out-of-fold predictions) from the rows on
    # which it is evaluated (predictions of models refitted on the outer training part).
    # "in_sample": selection and evaluation on the same out-of-fold predictions
    # (exploratory only; optimistic, see docs/manuscript_decisions.md).
    "protocol": "nested",
    "n_outer_folds": 5,
    # Feature columns removed before training (in addition to the automatically
    # removed constant and identifier columns, see data/loader.py).
    "drop_columns": None,
    # Model pool: per algorithm, 1 original model + (max_models_per_algo - 1)
    # feature-subset variants drawing 60-90 % of the features.
    "max_models_per_algo": 20,
    "feature_subset_min_ratio": 0.6,
    "feature_subset_max_ratio": 0.9,
    # "auto": macro-F1 (classification) / RMSE (regression)
    "performance_metric": "auto",
    # "auto": disagreement (classification) / ambiguity (regression)
    "diversity_metric": "auto",
    # Trade-off weights alpha for fitness = alpha * P(S) + (1 - alpha) * D(S).
    # None -> DEFAULT_ALPHA_GRID = {0.0, 0.1, ..., 1.0}
    "alpha_grid": None,
    # Rescaling of P(S) and D(S) before mixing: "minmax" (robust 1st/99th percentile
    # bounds from a reference sample of random subsets drawn with ``seed``) or "none".
    "fitness_normalization": "minmax",
    "fitness_reference_subsets": 500,
    # Fixed base-model hyperparameters and stacking settings (JSON).
    # None -> bundled prunestack/configs/hyperparameters.json
    "hyperparameters_path": None,
    # Stacking combiner; None -> method defined in the hyperparameters JSON.
    "stacking_method": None,
    # Level-0 inputs of stacking for classification: "probabilities" (out-of-fold
    # predict_proba, Ting & Witten 1999) or "labels" (predicted classes).
    # Regression always stacks the real-valued predictions.
    "stacking_inputs": "probabilities",
    # Elephant Herding Optimization ("standard" operators, see optimization/eho.py;
    # "legacy" reproduces PruneStack <= 2.0.0)
    "optimizer": "eho",
    "eho_variant": "standard",
    "eho_n_clans": 10,
    "eho_clan_size": 10,
    "eho_max_iterations": 100,
    "eho_alpha": 0.5,
    "eho_beta": 0.1,
    "eho_early_stop_patience": 25,
    # Per-elephant inclusion probability at initialisation ~ U(min, max)
    "eho_init_inclusion_min": 0.05,
    "eho_init_inclusion_max": 0.5,
    # Bit-flip probability per bit; None -> 1 / n_models
    "eho_mutation_rate": None,
    # Fitness-evaluation budget per optimization; None -> unlimited
    "eho_max_evaluations": None,
    # Seed of the optimizer; None -> same as ``seed``.
    "optimization_seed": None,
}

VALID_TASK_TYPES = {"auto", "classification", "regression"}
VALID_PROTOCOLS = {"nested", "in_sample"}
VALID_OPTIMIZERS = {"eho"}
VALID_EHO_VARIANTS = {"standard", "legacy"}
VALID_FITNESS_NORMALIZATIONS = {"minmax", "none"}
VALID_PERFORMANCE_METRICS = {"auto", "accuracy", "f1", "mae", "mse", "rmse", "r2"}
VALID_DIVERSITY_METRICS = {
    "auto",
    "disagreement", "q_statistic", "double_fault",  # classification
    "ambiguity", "correlation", "rmse_difference",  # regression
}


@dataclass
class ExperimentConfig:
    """Validated experiment configuration (see ``DEFAULTS`` for documentation)."""

    dataset_path: Path
    seed: int = DEFAULTS["seed"]
    n_folds: int = DEFAULTS["n_folds"]
    task_type: str = DEFAULTS["task_type"]
    protocol: str = DEFAULTS["protocol"]
    n_outer_folds: int = DEFAULTS["n_outer_folds"]
    drop_columns: list[str] | None = DEFAULTS["drop_columns"]
    max_models_per_algo: int = DEFAULTS["max_models_per_algo"]
    feature_subset_min_ratio: float = DEFAULTS["feature_subset_min_ratio"]
    feature_subset_max_ratio: float = DEFAULTS["feature_subset_max_ratio"]
    performance_metric: str = DEFAULTS["performance_metric"]
    diversity_metric: str = DEFAULTS["diversity_metric"]
    alpha_grid: list[float] | None = DEFAULTS["alpha_grid"]
    fitness_normalization: str = DEFAULTS["fitness_normalization"]
    fitness_reference_subsets: int = DEFAULTS["fitness_reference_subsets"]
    hyperparameters_path: Path | None = DEFAULTS["hyperparameters_path"]
    stacking_method: str | None = DEFAULTS["stacking_method"]
    stacking_inputs: str = DEFAULTS["stacking_inputs"]
    optimizer: str = DEFAULTS["optimizer"]
    eho_variant: str = DEFAULTS["eho_variant"]
    eho_n_clans: int = DEFAULTS["eho_n_clans"]
    eho_clan_size: int = DEFAULTS["eho_clan_size"]
    eho_max_iterations: int = DEFAULTS["eho_max_iterations"]
    eho_alpha: float = DEFAULTS["eho_alpha"]
    eho_beta: float = DEFAULTS["eho_beta"]
    eho_early_stop_patience: int = DEFAULTS["eho_early_stop_patience"]
    eho_init_inclusion_min: float = DEFAULTS["eho_init_inclusion_min"]
    eho_init_inclusion_max: float = DEFAULTS["eho_init_inclusion_max"]
    eho_mutation_rate: float | None = DEFAULTS["eho_mutation_rate"]
    eho_max_evaluations: int | None = DEFAULTS["eho_max_evaluations"]
    optimization_seed: int | None = DEFAULTS["optimization_seed"]

    def __post_init__(self) -> None:
        self.dataset_path = Path(self.dataset_path)
        if self.hyperparameters_path is not None:
            self.hyperparameters_path = Path(self.hyperparameters_path)
        if self.alpha_grid is not None:
            self.alpha_grid = [float(a) for a in self.alpha_grid]
        if self.drop_columns is not None:
            self.drop_columns = [str(c) for c in self.drop_columns]
        _validate_config(self)

    @property
    def resolved_alpha_grid(self) -> list[float]:
        """Alpha grid actually used by the experiment."""
        return list(self.alpha_grid) if self.alpha_grid is not None else list(DEFAULT_ALPHA_GRID)

    @property
    def resolved_optimization_seed(self) -> int:
        """Optimizer seed actually used by the experiment."""
        return self.seed if self.optimization_seed is None else self.optimization_seed

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (paths converted to strings)."""
        out: dict[str, Any] = {}
        for key in ("dataset_path", *DEFAULTS):
            value = getattr(self, key)
            out[key] = str(value) if isinstance(value, Path) else value
        return out


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_config(cfg: ExperimentConfig) -> None:
    """Raise ``ValueError`` / ``FileNotFoundError`` on invalid settings."""
    from prunestack.ensemble.stacking import META_LEARNERS  # local import avoids a cycle

    if not cfg.dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {cfg.dataset_path}")
    if cfg.hyperparameters_path is not None and not cfg.hyperparameters_path.is_file():
        raise FileNotFoundError(f"Hyperparameters file does not exist: {cfg.hyperparameters_path}")

    _check(cfg.seed >= 0, f"seed must be non-negative, got {cfg.seed}")
    _check(cfg.n_folds >= 2, f"n_folds must be >= 2, got {cfg.n_folds}")
    _check(cfg.task_type in VALID_TASK_TYPES,
           f"task_type must be one of {sorted(VALID_TASK_TYPES)}, got '{cfg.task_type}'")
    _check(cfg.protocol in VALID_PROTOCOLS,
           f"protocol must be one of {sorted(VALID_PROTOCOLS)}, got '{cfg.protocol}'")
    _check(cfg.n_outer_folds >= 2, f"n_outer_folds must be >= 2, got {cfg.n_outer_folds}")
    _check(cfg.max_models_per_algo >= 1,
           f"max_models_per_algo must be >= 1, got {cfg.max_models_per_algo}")
    _check(0.0 < cfg.feature_subset_min_ratio < 1.0,
           f"feature_subset_min_ratio must be in (0, 1), got {cfg.feature_subset_min_ratio}")
    _check(0.0 < cfg.feature_subset_max_ratio <= 1.0,
           f"feature_subset_max_ratio must be in (0, 1], got {cfg.feature_subset_max_ratio}")
    _check(cfg.feature_subset_min_ratio < cfg.feature_subset_max_ratio,
           f"feature_subset_min_ratio ({cfg.feature_subset_min_ratio}) must be < "
           f"feature_subset_max_ratio ({cfg.feature_subset_max_ratio})")
    _check(cfg.performance_metric in VALID_PERFORMANCE_METRICS,
           f"performance_metric must be one of {sorted(VALID_PERFORMANCE_METRICS)}, "
           f"got '{cfg.performance_metric}'")
    _check(cfg.diversity_metric in VALID_DIVERSITY_METRICS,
           f"diversity_metric must be one of {sorted(VALID_DIVERSITY_METRICS)}, "
           f"got '{cfg.diversity_metric}'")
    _check(cfg.stacking_method is None or cfg.stacking_method in META_LEARNERS,
           f"stacking_method must be one of {sorted(META_LEARNERS)}, got '{cfg.stacking_method}'")
    _check(cfg.stacking_inputs in ("probabilities", "labels"),
           f"stacking_inputs must be 'probabilities' or 'labels', got '{cfg.stacking_inputs}'")
    if cfg.alpha_grid is not None:
        _check(len(cfg.alpha_grid) > 0, "alpha_grid must not be empty")
        _check(all(0.0 <= a <= 1.0 for a in cfg.alpha_grid),
               "all values in alpha_grid must be in [0, 1]")
    _check(cfg.fitness_normalization in VALID_FITNESS_NORMALIZATIONS,
           f"fitness_normalization must be one of {sorted(VALID_FITNESS_NORMALIZATIONS)}, "
           f"got '{cfg.fitness_normalization}'")
    _check(cfg.fitness_reference_subsets >= 2,
           f"fitness_reference_subsets must be >= 2, got {cfg.fitness_reference_subsets}")

    _check(cfg.optimizer in VALID_OPTIMIZERS,
           f"optimizer must be one of {sorted(VALID_OPTIMIZERS)}, got '{cfg.optimizer}'")
    _check(cfg.eho_variant in VALID_EHO_VARIANTS,
           f"eho_variant must be one of {sorted(VALID_EHO_VARIANTS)}, got '{cfg.eho_variant}'")
    _check(cfg.eho_n_clans >= 1, f"eho_n_clans must be >= 1, got {cfg.eho_n_clans}")
    _check(cfg.eho_clan_size >= 2, f"eho_clan_size must be >= 2, got {cfg.eho_clan_size}")
    _check(cfg.eho_max_iterations >= 1,
           f"eho_max_iterations must be >= 1, got {cfg.eho_max_iterations}")
    _check(0.0 <= cfg.eho_alpha <= 1.0, f"eho_alpha must be in [0, 1], got {cfg.eho_alpha}")
    _check(0.0 <= cfg.eho_beta <= 1.0, f"eho_beta must be in [0, 1], got {cfg.eho_beta}")
    _check(cfg.eho_early_stop_patience >= 0,
           f"eho_early_stop_patience must be >= 0, got {cfg.eho_early_stop_patience}")
    _check(0.0 < cfg.eho_init_inclusion_min <= cfg.eho_init_inclusion_max < 1.0,
           "eho_init_inclusion_min/max must satisfy 0 < min <= max < 1, got "
           f"{cfg.eho_init_inclusion_min}, {cfg.eho_init_inclusion_max}")
    _check(cfg.eho_mutation_rate is None or 0.0 <= cfg.eho_mutation_rate <= 1.0,
           f"eho_mutation_rate must be in [0, 1], got {cfg.eho_mutation_rate}")
    _check(cfg.eho_max_evaluations is None or cfg.eho_max_evaluations >= 1,
           f"eho_max_evaluations must be >= 1, got {cfg.eho_max_evaluations}")
    _check(cfg.optimization_seed is None or cfg.optimization_seed >= 0,
           f"optimization_seed must be non-negative, got {cfg.optimization_seed}")


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file into a dictionary."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(raw).__name__}")
    return raw


def build_config(
    dataset_path: str | Path,
    config_path: str | Path | None = None,
    **overrides: Any,
) -> ExperimentConfig:
    """Merge defaults, an optional YAML file and keyword overrides (``None`` values ignored)."""
    merged: dict[str, Any] = dict(DEFAULTS)

    if config_path is not None:
        yaml_cfg = load_yaml_config(config_path)
        unknown = set(yaml_cfg) - set(DEFAULTS) - {"dataset_path"}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        merged.update({k: v for k, v in yaml_cfg.items() if k in DEFAULTS})

    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    merged.update({k: v for k, v in overrides.items() if v is not None})

    return ExperimentConfig(dataset_path=Path(dataset_path), **merged)
