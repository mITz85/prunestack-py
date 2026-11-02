"""Fixed base-model hyperparameters and stacking settings from a public JSON file.

All base learners use scikit-learn default hyperparameters, with minor
adjustments to iteration limits to ensure convergence.  The configuration is
identical for every model in the pool (original model and all feature-subset
variants); no per-dataset hyperparameter optimisation is performed.

Resolution order for the configuration file:

1. an explicit ``config_path`` (e.g. ``--hyperparameters PATH``);
2. the repository-level ``configs/hyperparameters.json`` (repo clone);
3. the copy bundled with the installed package
   (``prunestack/configs/hyperparameters.json``, via :mod:`importlib.resources`).
"""

from __future__ import annotations

import copy
import json
import logging
from importlib import resources
from pathlib import Path
from typing import Any

from prunestack.models.registry import (
    CLASSIFICATION_ALGORITHMS,
    REGRESSION_ALGORITHMS,
    get_model_class,
)

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "hyperparameters.json"

# <repo>/src/prunestack/models/hyperparameters.py -> <repo>/configs/
_REPO_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / CONFIG_FILENAME

_REGISTRY_ALGORITHMS = {
    "classification": CLASSIFICATION_ALGORITHMS,
    "regression": REGRESSION_ALGORITHMS,
}


def _check_task_type(task_type: str) -> None:
    if task_type not in _REGISTRY_ALGORITHMS:
        raise ValueError(
            f"task_type must be 'classification' or 'regression', got {task_type!r}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Hyperparameter config {path} must contain a JSON object")
    return data


def _load_raw_config(config_path: str | Path | None = None) -> tuple[dict[str, Any], str]:
    """Load the raw JSON config following the documented resolution order.

    Returns the parsed dict and a human-readable description of its source.
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Hyperparameters file does not exist: {path}")
        return _read_json(path), str(path)

    if _REPO_CONFIG_PATH.is_file():
        return _read_json(_REPO_CONFIG_PATH), str(_REPO_CONFIG_PATH)

    resource = resources.files("prunestack").joinpath("configs", CONFIG_FILENAME)
    if not resource.is_file():
        raise FileNotFoundError(
            "No hyperparameter configuration found: neither "
            f"{_REPO_CONFIG_PATH} nor the packaged prunestack/configs/{CONFIG_FILENAME}"
        )
    data = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Packaged hyperparameter config must contain a JSON object")
    return data, f"package resource prunestack/configs/{CONFIG_FILENAME}"


def _public_items(section: dict[str, Any]) -> dict[str, Any]:
    """Drop metadata keys (those starting with '_')."""
    return {k: v for k, v in section.items() if not str(k).startswith("_")}


def load_hyperparameters(
    task_type: str,
    seed: int,
    config_path: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Load fixed hyperparameters for all base learners of a task.

    Parameters
    ----------
    task_type : 'classification' | 'regression'
    seed : int
        Experiment seed, injected as ``random_state`` for every algorithm that
        accepts it.
    config_path : str or Path or None
        Optional custom JSON file; see the module docstring for the fallback order.

    Returns
    -------
    dict[str, dict[str, Any]]
        Algorithm name -> parameter dict, in registry order.  Returned dicts are
        fresh copies; the loaded configuration is never mutated.

    Raises
    ------
    ValueError
        If the task section is missing, or its algorithm names do not exactly
        match the registry set for the task (missing or unknown names).
    """
    _check_task_type(task_type)
    raw, source = _load_raw_config(config_path)

    section = raw.get(task_type)
    if not isinstance(section, dict):
        raise ValueError(
            f"Hyperparameter config ({source}) has no '{task_type}' section"
        )
    section = _public_items(section)

    expected = list(_REGISTRY_ALGORITHMS[task_type])
    missing = sorted(set(expected) - set(section))
    unknown = sorted(set(section) - set(expected))
    if missing or unknown:
        raise ValueError(
            f"Hyperparameter config ({source}) section '{task_type}' must list "
            f"exactly the registry algorithms {expected}. "
            f"Missing: {missing or 'none'}; unknown: {unknown or 'none'}."
        )

    result: dict[str, dict[str, Any]] = {}
    for algo in expected:
        params = section[algo]
        if not isinstance(params, dict):
            raise ValueError(
                f"Hyperparameters for '{algo}' ({source}) must be a JSON object, "
                f"got {type(params).__name__}"
            )
        params = copy.deepcopy(params)
        model_cls = get_model_class(algo, task_type)
        if "random_state" in model_cls().get_params():
            params["random_state"] = seed
        result[algo] = params

    logger.info(
        "Loaded fixed hyperparameters for %d %s algorithms from %s",
        len(result), task_type, source,
    )
    return result


def load_stacking_config(
    task_type: str,
    config_path: str | Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """Return ``(meta_learner, params)`` from the ``stacking`` section of the JSON.

    Falls back to ``("nnls_mlr", {})`` for classification and ``("nnls", {})`` for
    regression when the section or the task entry is missing.
    """
    from prunestack.ensemble.stacking import DEFAULT_META_LEARNER

    _check_task_type(task_type)
    raw, source = _load_raw_config(config_path)
    entry = (raw.get("stacking") or {}).get(task_type) or {}
    if not isinstance(entry, dict):
        raise ValueError(f"stacking.{task_type} ({source}) must be a JSON object")
    meta_learner = entry.get("meta_learner", DEFAULT_META_LEARNER[task_type])
    params = entry.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"stacking.{task_type}.params ({source}) must be a JSON object")
    unknown = set(_public_items(entry)) - {"meta_learner", "params"}
    if unknown:
        raise ValueError(f"stacking.{task_type} ({source}) has unknown keys: {sorted(unknown)}")
    logger.info("Stacking meta-learner for %s: %s %s (%s)", task_type, meta_learner, params, source)
    return meta_learner, copy.deepcopy(params)
