"""Tests for task type detection."""

import pytest
import pandas as pd

from prunestack.data.task_detection import detect_task_type


class TestDetectTaskType:
    """Tests for detect_task_type function."""

    def test_classification_detected(self, classification_metrics_df):
        assert detect_task_type(classification_metrics_df) == "classification"

    def test_regression_detected(self, regression_metrics_df):
        assert detect_task_type(regression_metrics_df) == "regression"

    def test_override_classification(self, regression_metrics_df):
        """Override should take precedence over auto-detection."""
        result = detect_task_type(regression_metrics_df, override="classification")
        assert result == "classification"

    def test_override_regression(self, classification_metrics_df):
        result = detect_task_type(classification_metrics_df, override="regression")
        assert result == "regression"

    def test_ambiguous_raises(self):
        """Both classification and regression columns present."""
        df = pd.DataFrame({
            "accuracy": [0.75],
            "mae": [2.5],
        })
        with pytest.raises(ValueError, match="Ambiguous"):
            detect_task_type(df)

    def test_no_indicators_raises(self):
        """No recognized metric columns."""
        df = pd.DataFrame({
            "unknown_metric": [0.5],
            "time": [1.0],
        })
        with pytest.raises(ValueError, match="Cannot detect"):
            detect_task_type(df)

    def test_invalid_override_raises(self):
        df = pd.DataFrame({"accuracy": [0.75]})
        with pytest.raises(ValueError, match="Invalid task_type"):
            detect_task_type(df, override="invalid")

    def test_case_insensitive_columns(self):
        """Column names should be case-insensitive."""
        df = pd.DataFrame({
            "Accuracy": [0.75],
            "F1": [0.74],
            "Precision": [0.76],
            "Recall": [0.75],
        })
        assert detect_task_type(df) == "classification"

    def test_classification_metrics_file(self, classification_dataset_dir):
        metrics_df = pd.read_csv(classification_dataset_dir / "metrics.csv", index_col=0)
        assert detect_task_type(metrics_df) == "classification"

    def test_regression_metrics_file(self, regression_dataset_dir):
        metrics_df = pd.read_csv(regression_dataset_dir / "metrics.csv", index_col=0)
        assert detect_task_type(metrics_df) == "regression"

    def test_partial_classification_indicators(self):
        """Even one classification indicator should trigger detection."""
        df = pd.DataFrame({"accuracy": [0.75], "time": [1.0]})
        assert detect_task_type(df) == "classification"

    def test_partial_regression_indicators(self):
        """Even one regression indicator should trigger detection."""
        df = pd.DataFrame({"mse": [10.0], "time": [1.0]})
        assert detect_task_type(df) == "regression"
