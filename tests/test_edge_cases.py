"""Edge case tests for the prunestack framework."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prunestack.data.validation import validate_dataset
from prunestack.metrics.fitness import calculate_fitness as compute_fitness


class TestSmallDatasets:
    """Test behavior with very small datasets."""

    def test_tiny_dataset_validation(self):
        X = np.random.rand(5, 3)
        y = np.array([0, 1, 0, 1, 0])
        result = validate_dataset(X, y)
        assert result.is_valid
        assert any("small" in w.lower() for w in result.warnings)

    def test_two_sample_dataset(self):
        X = np.random.rand(2, 3)
        y = np.array([0, 1])
        result = validate_dataset(X, y)
        assert result.is_valid  # valid but with warnings

    def test_single_feature(self):
        X = np.random.rand(50, 1)
        y = np.random.randint(0, 2, 50)
        result = validate_dataset(X, y)
        assert result.is_valid


class TestBoundaryConditions:
    """Test boundary conditions for optimization parameters."""

    def test_alpha_zero(self):
        """Alpha=0: pure diversity, no performance weight."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pred = np.column_stack([
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1]),
        ]).astype(float)
        fitness = compute_fitness(y, pred, [0, 1, 2], alpha=0.0, task_type="classification")
        assert isinstance(fitness, float)

    def test_alpha_one(self):
        """Alpha=1: pure performance, no diversity weight."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pred = np.column_stack([
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1]),
        ]).astype(float)
        fitness = compute_fitness(y, pred, [0, 1, 2], alpha=1.0, task_type="classification")
        assert isinstance(fitness, float)

    def test_alpha_half_complementary(self):
        """alpha=0.5 weights performance and diversity equally."""
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        pred = np.column_stack([
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        ]).astype(float)
        fitness = compute_fitness(y, pred, [0, 1], alpha=0.5, task_type="classification")
        assert isinstance(fitness, float)

    def test_single_model_subset(self):
        """Single model in subset."""
        y = np.array([0, 1, 0, 1, 0])
        pred = np.column_stack([
            np.array([0, 1, 0, 1, 0]),
            np.array([1, 0, 1, 0, 1]),
        ]).astype(float)
        fitness = compute_fitness(y, pred, [0], alpha=0.5, task_type="classification")
        assert isinstance(fitness, float)

    def test_all_models_subset(self):
        """All models in subset."""
        y = np.array([0, 1, 0, 1, 0])
        pred = np.column_stack([
            np.array([0, 1, 0, 1, 0]),
            np.array([1, 0, 1, 0, 1]),
            np.array([0, 0, 1, 1, 0]),
        ]).astype(float)
        fitness = compute_fitness(y, pred, [0, 1, 2], alpha=0.5, task_type="classification")
        assert isinstance(fitness, float)


class TestClassImbalance:
    """Test handling of class imbalance."""

    def test_binary_balanced(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 50 + [1] * 50)
        result = validate_dataset(X, y)
        assert not any("imbalance" in w.lower() for w in result.warnings)

    def test_binary_moderate_imbalance(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 80 + [1] * 20)
        result = validate_dataset(X, y)
        assert any("imbalance" in w.lower() for w in result.warnings)

    def test_binary_severe_imbalance(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 98 + [1] * 2)
        result = validate_dataset(X, y)
        assert any("severe" in w.lower() for w in result.warnings)
