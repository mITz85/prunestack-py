"""Deterministic feature subset generation for base model diversity."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_feature_subsets(
    n_features: int,
    n_variants: int = 19,
    base_seed: int = 42,
    algo_idx: int = 0,
    min_ratio: float = 0.6,
    max_ratio: float = 0.9,
) -> list[list[int]]:
    """Generate deterministic random feature subsets for one algorithm.

    Each variant gets a unique seed: base_seed + algo_idx * 1000 + variant_idx.
    Subset sizes are sampled uniformly from [min_ratio * n_features, max_ratio * n_features].
    Feature indices are sampled without replacement.
    Duplicate subsets are rejected and resampled.

    Parameters
    ----------
    n_features : int
        Total number of features in the dataset.
    n_variants : int
        Number of feature-subset variants (default 19; together with the original
        model this gives 20 models per algorithm).
    base_seed : int
        Global random seed.
    algo_idx : int
        Algorithm index (0-based) for seed derivation.
    min_ratio : float
        Minimum fraction of features to include.
    max_ratio : float
        Maximum fraction of features to include.

    Returns
    -------
    list[list[int]]
        List of sorted feature index lists, one per variant.
    """
    if n_features < 2:
        logger.warning(
            "Too few features (%d) for subset generation; returning empty list.",
            n_features,
        )
        return []

    min_size = max(1, int(min_ratio * n_features))
    max_size = max(min_size, int(max_ratio * n_features))

    # Clamp max_size to n_features - 1 to ensure subsets differ from all-features
    max_size = min(max_size, n_features - 1)

    if min_size > max_size:
        min_size = max_size

    subsets: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    max_attempts = n_variants * 10  # retry budget for duplicates
    attempts = 0

    variant_idx = 0
    while len(subsets) < n_variants and attempts < max_attempts:
        seed = base_seed + algo_idx * 1000 + variant_idx
        rng = np.random.RandomState(seed)

        subset_size = rng.randint(min_size, max_size + 1)
        indices = sorted(rng.choice(n_features, size=subset_size, replace=False).tolist())
        key = tuple(indices)

        if key not in seen:
            seen.add(key)
            subsets.append(indices)
        else:
            logger.debug(
                "Duplicate subset at variant_idx=%d for algo_idx=%d, skipping.",
                variant_idx, algo_idx,
            )

        variant_idx += 1
        attempts += 1

    if len(subsets) < n_variants:
        logger.warning(
            "Could only generate %d/%d unique subsets for algo_idx=%d "
            "(n_features=%d). Dataset may be too small.",
            len(subsets), n_variants, algo_idx, n_features,
        )

    return subsets
