"""Custom exception classes for the prunestack framework.

Provides fine-grained error types with actionable messages so that
users can diagnose and fix problems quickly.
"""

from __future__ import annotations


class PruneStackError(Exception):
    """Base exception for all prunestack errors."""

    def __init__(self, message: str, suggestion: str | None = None) -> None:
        self.suggestion = suggestion
        full = message
        if suggestion:
            full += f"\n  💡 Suggestion: {suggestion}"
        super().__init__(full)


class DataValidationError(PruneStackError):
    """Raised when input data fails validation checks."""


class ConfigurationError(PruneStackError):
    """Raised when experiment configuration is invalid."""


class OptimizationError(PruneStackError):
    """Raised when the optimization process fails."""


class ModelTrainingError(PruneStackError):
    """Raised when model training encounters an unrecoverable error."""


class InsufficientDataError(DataValidationError):
    """Raised when there is not enough data for the requested operation."""
