"""Tests for error handling throughout the pipeline."""

from __future__ import annotations

import pytest

from prunestack.utils.exceptions import (
    PruneStackError,
    DataValidationError,
    ConfigurationError,
)
from prunestack.data.validation import run_all_validations, validate_dataset
import numpy as np


class TestErrorMessages:
    """Test that error messages are clear and actionable."""

    def test_empty_dataset_error_message(self):
        X = np.empty((0, 5))
        y = np.empty(0)
        result = validate_dataset(X, y)
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_shape_mismatch_error_message(self):
        X = np.random.rand(10, 5)
        y = np.arange(15)
        result = validate_dataset(X, y)
        assert not result.is_valid
        assert any("mismatch" in e.lower() for e in result.errors)

    def test_config_error_message(self):
        from prunestack.config import ExperimentConfig
        with pytest.raises((ValueError, FileNotFoundError)):
            ExperimentConfig(dataset_path="/nonexistent/path")

    def test_suggestion_in_exceptions(self):
        err = DataValidationError(
            "Validation failed",
            suggestion="Check your data format",
        )
        msg = str(err)
        assert "Suggestion" in msg
        assert "Check your data format" in msg

    def test_prunestack_error_is_catchable(self):
        """All custom exceptions should be catchable as PruneStackError."""
        exceptions = [
            DataValidationError("test"),
            ConfigurationError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, PruneStackError)


class TestStrictValidation:
    """Test strict validation mode."""

    def test_strict_converts_warnings_to_errors(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(dataset_path=str(tiny_dataset_dir))
        # Small dataset produces warnings
        X = np.random.rand(10, 5)
        y = np.arange(10)
        hyperparameters = {"DecisionTreeClassifier": {"max_depth": 5}}

        with pytest.raises(DataValidationError):
            run_all_validations(X, y, hyperparameters, config, strict=True)

    def test_non_strict_allows_warnings(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(dataset_path=str(tiny_dataset_dir))
        X = np.random.rand(10, 5)
        y = np.arange(10)
        hyperparameters = {"DecisionTreeClassifier": {"max_depth": 5}}

        # Should not raise (only warnings, not errors)
        result = run_all_validations(X, y, hyperparameters, config, strict=False)
        assert result.is_valid
