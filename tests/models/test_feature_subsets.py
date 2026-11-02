"""Tests for feature subset generation."""

import pytest

from prunestack.models.feature_subsets import generate_feature_subsets


class TestGenerateFeatureSubsets:
    def test_basic_generation(self):
        subsets = generate_feature_subsets(n_features=20, n_variants=9, base_seed=42, algo_idx=0)
        assert len(subsets) == 9
        for subset in subsets:
            assert len(subset) > 0
            assert all(0 <= idx < 20 for idx in subset)
            assert subset == sorted(subset)  # sorted

    def test_deterministic(self):
        s1 = generate_feature_subsets(n_features=20, n_variants=9, base_seed=42, algo_idx=0)
        s2 = generate_feature_subsets(n_features=20, n_variants=9, base_seed=42, algo_idx=0)
        assert s1 == s2

    def test_different_seed_different_subsets(self):
        s1 = generate_feature_subsets(n_features=20, n_variants=9, base_seed=42, algo_idx=0)
        s2 = generate_feature_subsets(n_features=20, n_variants=9, base_seed=99, algo_idx=0)
        assert s1 != s2

    def test_different_algo_idx_different_subsets(self):
        s1 = generate_feature_subsets(n_features=20, n_variants=9, base_seed=42, algo_idx=0)
        s2 = generate_feature_subsets(n_features=20, n_variants=9, base_seed=42, algo_idx=1)
        assert s1 != s2

    def test_subset_sizes_within_ratio(self):
        subsets = generate_feature_subsets(
            n_features=40, n_variants=9, base_seed=42, algo_idx=0,
            min_ratio=0.5, max_ratio=0.9,
        )
        for subset in subsets:
            assert len(subset) >= int(0.5 * 40)
            assert len(subset) <= int(0.9 * 40)

    def test_no_duplicate_subsets(self):
        subsets = generate_feature_subsets(n_features=20, n_variants=9, base_seed=42, algo_idx=0)
        subset_tuples = [tuple(s) for s in subsets]
        assert len(set(subset_tuples)) == len(subset_tuples)

    def test_no_duplicate_indices_in_subset(self):
        subsets = generate_feature_subsets(n_features=20, n_variants=9, base_seed=42, algo_idx=0)
        for subset in subsets:
            assert len(set(subset)) == len(subset)

    def test_small_feature_space(self):
        """With very few features, should generate as many as possible."""
        subsets = generate_feature_subsets(
            n_features=3, n_variants=9, base_seed=42, algo_idx=0,
            min_ratio=0.5, max_ratio=0.9,
        )
        # May not get all 9 unique subsets with only 3 features
        assert len(subsets) <= 9
        assert len(subsets) > 0

    def test_single_feature(self):
        """Edge case: only 1 feature, can't generate subsets."""
        subsets = generate_feature_subsets(n_features=1, n_variants=9, base_seed=42, algo_idx=0)
        assert subsets == []

    def test_seed_formula(self):
        """Verify seed = base_seed + algo_idx * 1000 + variant_idx."""
        import numpy as np
        # First variant of algo_idx=2, base_seed=42 should use seed=2042
        rng = np.random.RandomState(42 + 2 * 1000 + 0)  # seed=2042
        subsets = generate_feature_subsets(
            n_features=20, n_variants=1, base_seed=42, algo_idx=2,
            min_ratio=0.5, max_ratio=0.9,
        )
        # Reproduce manually
        min_size = int(0.5 * 20)
        max_size = min(int(0.9 * 20), 19)
        subset_size = rng.randint(min_size, max_size + 1)
        expected = sorted(rng.choice(20, size=subset_size, replace=False).tolist())
        assert subsets[0] == expected

    def test_four_algos_forty_features(self):
        """Simulates real scenario: 4 algos × 9 variants on 40-feature dataset."""
        all_subsets = []
        for algo_idx in range(4):
            subsets = generate_feature_subsets(
                n_features=40, n_variants=9, base_seed=42, algo_idx=algo_idx,
            )
            assert len(subsets) == 9
            all_subsets.extend(subsets)
        # Total should be 36 subsets
        assert len(all_subsets) == 36
