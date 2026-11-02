"""Tests for prunestack.utils.exceptions module."""

from __future__ import annotations

import pytest

from prunestack.utils.exceptions import (
    PruneStackError,
    DataValidationError,
    ConfigurationError,
    OptimizationError,
    ModelTrainingError,
    InsufficientDataError,
)


class TestExceptionHierarchy:
    """Test custom exception hierarchy."""

    def test_base_exception(self):
        with pytest.raises(PruneStackError):
            raise PruneStackError("test error")

    def test_suggestion_message(self):
        err = PruneStackError("bad thing", suggestion="try this")
        assert "Suggestion" in str(err)
        assert "try this" in str(err)
        assert err.suggestion == "try this"

    def test_no_suggestion(self):
        err = PruneStackError("bad thing")
        assert err.suggestion is None
        assert "Suggestion" not in str(err)

    def test_data_validation_inherits(self):
        with pytest.raises(PruneStackError):
            raise DataValidationError("data bad")

    def test_configuration_inherits(self):
        with pytest.raises(PruneStackError):
            raise ConfigurationError("config bad")

    def test_optimization_inherits(self):
        with pytest.raises(PruneStackError):
            raise OptimizationError("opt failed")

    def test_model_training_inherits(self):
        with pytest.raises(PruneStackError):
            raise ModelTrainingError("train failed")

    def test_insufficient_data_inherits(self):
        # Should be catchable as both DataValidationError and PruneStackError
        with pytest.raises(DataValidationError):
            raise InsufficientDataError("not enough data")
        with pytest.raises(PruneStackError):
            raise InsufficientDataError("not enough data")

    def test_all_have_suggestion(self):
        for exc_class in [
            DataValidationError,
            ConfigurationError,
            OptimizationError,
            ModelTrainingError,
        ]:
            err = exc_class("msg", suggestion="fix it")
            assert "fix it" in str(err)
