"""Tests for prunestack.data.validation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prunestack.data.validation import (
    ValidationResult,
    validate_dataset,
    validate_hyperparameters,
    validate_config,
    run_all_validations,
)
from prunestack.utils.exceptions import DataValidationError


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_initial_state(self):
        vr = ValidationResult()
        assert vr.is_valid
        assert vr.errors == []
        assert vr.warnings == []
        assert vr.info == []

    def test_add_error(self):
        vr = ValidationResult()
        vr.add_error("something broke")
        assert not vr.is_valid
        assert len(vr.errors) == 1

    def test_add_warning(self):
        vr = ValidationResult()
        vr.add_warning("caution")
        assert vr.is_valid  # warnings don't invalidate
        assert len(vr.warnings) == 1

    def test_add_info(self):
        vr = ValidationResult()
        vr.add_info("FYI")
        assert vr.is_valid
        assert len(vr.info) == 1

    def test_merge(self):
        vr1 = ValidationResult()
        vr1.add_warning("w1")
        vr2 = ValidationResult()
        vr2.add_error("e1")
        vr1.merge(vr2)
        assert not vr1.is_valid
        assert len(vr1.warnings) == 1
        assert len(vr1.errors) == 1

    def test_summary_clean(self):
        vr = ValidationResult()
        assert "passed" in vr.summary().lower()

    def test_summary_with_issues(self):
        vr = ValidationResult()
        vr.add_error("e")
        vr.add_warning("w")
        s = vr.summary()
        assert "1 error" in s
        assert "1 warning" in s


class TestValidateDataset:
    """Tests for the validate_dataset function."""

    def test_valid_classification(self):
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 3, 100)
        result = validate_dataset(X, y)
        assert result.is_valid

    def test_valid_regression(self):
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        result = validate_dataset(X, y)
        assert result.is_valid

    def test_empty_dataset(self):
        X = np.empty((0, 5))
        y = np.empty(0)
        result = validate_dataset(X, y)
        assert not result.is_valid
        assert any("empty" in e.lower() for e in result.errors)

    def test_no_features(self):
        X = np.empty((10, 0))
        y = np.arange(10)
        result = validate_dataset(X, y)
        assert not result.is_valid

    def test_shape_mismatch(self):
        X = np.random.rand(10, 5)
        y = np.arange(15)
        result = validate_dataset(X, y)
        assert not result.is_valid
        assert any("mismatch" in e.lower() for e in result.errors)

    def test_missing_values_warning(self):
        X = np.random.rand(50, 5)
        X[0, 0] = np.nan
        y = np.arange(50)
        result = validate_dataset(X, y)
        assert result.is_valid  # warning, not error
        assert any("missing" in w.lower() for w in result.warnings)

    def test_constant_features_warning(self):
        X = np.random.rand(50, 5)
        X[:, 2] = 1.0  # constant feature
        y = np.arange(50)
        result = validate_dataset(X, y)
        assert any("constant" in w.lower() for w in result.warnings)

    def test_high_dimensional_warning(self):
        X = np.random.rand(10, 50)  # more features than samples
        y = np.arange(10)
        result = validate_dataset(X, y)
        assert any("high-dimensional" in w.lower() for w in result.warnings)

    def test_small_dataset_warning(self):
        X = np.random.rand(15, 5)
        y = np.arange(15)
        result = validate_dataset(X, y)
        assert any("small" in w.lower() for w in result.warnings)

    def test_class_imbalance_warning(self):
        X = np.random.rand(100, 5)
        y = np.array([0] * 95 + [1] * 5)  # severe imbalance
        result = validate_dataset(X, y)
        assert any("imbalance" in w.lower() for w in result.warnings)

    def test_pandas_input(self):
        X = pd.DataFrame(np.random.rand(50, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.Series(np.random.randint(0, 2, 50))
        result = validate_dataset(X, y)
        assert result.is_valid

    def test_duplicate_samples_detected(self):
        X = np.vstack([np.ones((5, 3)), np.random.rand(45, 3)])
        y = np.arange(50)
        result = validate_dataset(X, y)
        # Should detect duplicates (the 5 identical rows)
        assert any("duplicate" in w.lower() for w in result.warnings)


class TestValidateHyperparameters:
    FULL_CLASSIFICATION = {
        "DecisionTreeClassifier": {}, "KNeighborsClassifier": {}, "LogisticRegression": {"max_iter": 1000},
        "MLPClassifier": {"max_iter": 1000}, "HistGradientBoostingClassifier": {"max_iter": 300},
        "RandomForestClassifier": {"n_estimators": 100},
    }

    def test_full_pool_is_clean(self):
        result = validate_hyperparameters(self.FULL_CLASSIFICATION, "classification")
        assert result.is_valid
        assert result.warnings == []

    def test_empty(self):
        result = validate_hyperparameters({}, "classification")
        assert not result.is_valid
        assert any("no base-learner hyperparameters" in e.lower() for e in result.errors)

    def test_single_algorithm_warns(self):
        result = validate_hyperparameters({"DecisionTreeClassifier": {}}, "classification")
        assert result.is_valid
        assert any("only 1" in w.lower() for w in result.warnings)
        assert any("fewer than 6" in w.lower() for w in result.warnings)

    @pytest.mark.parametrize("algorithm", ["MyCustomAlgo", "LinearSVC", "RidgeClassifier", "Ridge"])
    def test_unregistered_algorithm_is_error(self, algorithm):
        result = validate_hyperparameters({algorithm: {}}, "classification")
        assert not result.is_valid

    def test_non_dict_params_is_error(self):
        result = validate_hyperparameters({"DecisionTreeClassifier": [1, 2]}, "classification")
        assert not result.is_valid

    def test_none_value_warns(self):
        result = validate_hyperparameters({"DecisionTreeClassifier": {"max_depth": None}}, "classification")
        assert any("none" in w.lower() for w in result.warnings)


class TestValidateConfig:
    """Tests for the validate_config function."""

    def test_valid_config(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(dataset_path=str(tiny_dataset_dir))
        result = validate_config(config)
        assert result.is_valid

    def test_invalid_alpha_grid(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(
            dataset_path=str(tiny_dataset_dir),
            alpha_grid=[0.0, 0.5, 1.0],
        )
        result = validate_config(config)
        assert result.is_valid  # within range

    def test_high_folds_warning(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(
            dataset_path=str(tiny_dataset_dir),
            n_folds=15,
        )
        result = validate_config(config)
        assert any("unusually high" in w.lower() for w in result.warnings)

    def test_eho_info(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(dataset_path=str(tiny_dataset_dir))
        result = validate_config(config)
        assert any("population" in i.lower() for i in result.info)


class TestRunAllValidations:
    """Tests for the run_all_validations convenience function."""

    def test_valid_run(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(dataset_path=str(tiny_dataset_dir))
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        hyperparameters = {"DecisionTreeClassifier": {"max_depth": 5}}

        result = run_all_validations(X, y, hyperparameters, config)
        assert result.is_valid

    def test_strict_mode_raises(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(dataset_path=str(tiny_dataset_dir))
        X = np.random.rand(15, 5)  # small => warning
        y = np.arange(15)
        hyperparameters = {"DecisionTreeClassifier": {"max_depth": 5}}

        with pytest.raises(DataValidationError):
            run_all_validations(X, y, hyperparameters, config, strict=True)

    def test_error_raises(self, tiny_dataset_dir):
        from prunestack.config import ExperimentConfig
        config = ExperimentConfig(dataset_path=str(tiny_dataset_dir))
        X = np.empty((0, 5))
        y = np.empty(0)
        hyperparameters = {"DecisionTreeClassifier": {"max_depth": 5}}

        with pytest.raises(DataValidationError):
            run_all_validations(X, y, hyperparameters, config)
