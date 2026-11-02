"""Tests for categorical feature handling in loader and validation.

Covers:
  - Detection and encoding of string/object feature columns in X
  - feature_encoders stored in DatasetBundle
  - Inverse transform capability
  - Mixed numeric + categorical features
  - All-categorical features with string labels
  - Validation module handling of categorical columns
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "tiny_dataset"


def _make_cat_dataset(tmp_path, X_data, y_data, metrics_type="classification"):
    """Create a minimal dataset dir with given X, y data."""
    d = tmp_path / "ds"
    d.mkdir(parents=True)

    X_data.to_csv(d / "X_test.csv")
    y_data.to_csv(d / "y_test.csv")

    if metrics_type == "classification":
        metrics = pd.DataFrame(
            {"accuracy": [0.9, 0.85], "f1": [0.88, 0.83]},
            index=["DecisionTreeClassifier", "KNeighborsClassifier"],
        )
    else:
        metrics = pd.DataFrame(
            {"mae": [0.5, 0.6], "r2": [0.9, 0.85]},
            index=["DecisionTreeRegressor", "Ridge"],
        )
    metrics.to_csv(d / "metrics.csv")
    return d


class TestCategoricalFeatureEncoding:
    """Tests for categorical feature detection and encoding in loader."""

    def test_all_string_features_encoded(self, tmp_path):
        """All-string feature columns should be label-encoded."""
        from prunestack.data.loader import load_dataset

        X = pd.DataFrame({
            "gender": ["Male", "Female", "Male", "Female"] * 5,
            "color": ["Red", "Blue", "Green", "Red"] * 5,
        })
        y = pd.Series([0, 1, 0, 1] * 5, name="target")
        ds = _make_cat_dataset(tmp_path, X, y)
        bundle = load_dataset(ds)

        # All features should now be numeric
        assert bundle.X.dtypes.apply(lambda dt: np.issubdtype(dt, np.integer)).all()
        assert len(bundle.feature_encoders) == 2
        assert "gender" in bundle.feature_encoders
        assert "color" in bundle.feature_encoders

    def test_mixed_numeric_and_string_features(self, tmp_path):
        """Only string columns should be encoded; numeric columns untouched."""
        from prunestack.data.loader import load_dataset

        X = pd.DataFrame({
            "age": [25.0, 30.0, 35.0, 40.0] * 5,
            "city": ["NYC", "LA", "NYC", "LA"] * 5,
            "score": [100, 200, 150, 250] * 5,
        })
        y = pd.Series([0, 1, 0, 1] * 5, name="target")
        ds = _make_cat_dataset(tmp_path, X, y)
        bundle = load_dataset(ds)

        assert len(bundle.feature_encoders) == 1
        assert "city" in bundle.feature_encoders
        # Numeric columns should be unchanged
        assert bundle.X["age"].dtype in [np.float64, float]
        assert bundle.X["score"].dtype in [np.int64, int]

    def test_feature_encoder_inverse_transform(self, tmp_path):
        """Stored encoders should support inverse transform."""
        from prunestack.data.loader import load_dataset

        X = pd.DataFrame({
            "animal": ["cat", "dog", "bird", "cat"] * 5,
        })
        y = pd.Series([0, 1, 0, 1] * 5, name="target")
        ds = _make_cat_dataset(tmp_path, X, y)
        bundle = load_dataset(ds)

        enc = bundle.feature_encoders["animal"]
        original = enc.inverse_transform(bundle.X["animal"].values)
        assert set(original) == {"cat", "dog", "bird"}

    def test_no_categorical_features_empty_encoders(self, tmp_path):
        """Pure numeric dataset should have empty feature_encoders."""
        from prunestack.data.loader import load_dataset

        X = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0, 4.0] * 5,
            "f2": [5.0, 6.0, 7.0, 8.0] * 5,
        })
        y = pd.Series([0, 1, 0, 1] * 5, name="target")
        ds = _make_cat_dataset(tmp_path, X, y)
        bundle = load_dataset(ds)

        assert bundle.feature_encoders == {}
        assert bundle.X.shape[1] == 2

    def test_existing_tiny_dataset_no_regression(self):
        """Loading tiny_dataset (all numeric) should still work with new field."""
        from prunestack.data.loader import load_dataset

        bundle = load_dataset(TINY)
        assert hasattr(bundle, "feature_encoders")
        assert bundle.feature_encoders == {}
        assert bundle.n_features == 5

    def test_many_categories_encoded(self, tmp_path):
        """Features with many unique string values should be encoded."""
        from prunestack.data.loader import load_dataset

        # 10 categories over 20 rows: high-cardinality but not an identifier.
        names = [f"name_{i % 10}" for i in range(20)]
        X = pd.DataFrame({"name": names, "num": np.arange(20.0)})
        y = pd.Series([i % 2 for i in range(20)], name="target")
        ds = _make_cat_dataset(tmp_path, X, y)
        bundle = load_dataset(ds)

        assert "name" in bundle.feature_encoders
        assert bundle.X["name"].max() == 9

    def test_identifier_column_dropped(self, tmp_path):
        """All-unique string columns are identifiers and are removed before encoding."""
        from prunestack.data.loader import load_dataset

        X = pd.DataFrame({"name": [f"n{i}" for i in range(20)], "num": np.arange(20.0)})
        y = pd.Series([i % 2 for i in range(20)], name="target")
        bundle = load_dataset(_make_cat_dataset(tmp_path, X, y))

        assert bundle.dropped_columns == {"name": "identifier"}
        assert bundle.feature_names == ["num"]

    def test_encoding_preserves_sample_count(self, tmp_path):
        """Encoding should not change number of samples."""
        from prunestack.data.loader import load_dataset

        X = pd.DataFrame({
            "a": ["x", "y", "z"] * 10,
            "b": [1.0, 2.0, 3.0] * 10,
        })
        y = pd.Series([0, 1, 0] * 10, name="target")
        ds = _make_cat_dataset(tmp_path, X, y)
        bundle = load_dataset(ds)

        assert bundle.n_samples == 30
        assert len(bundle.X) == 30

    def test_encoding_deterministic(self, tmp_path):
        """Same data should produce same encodings."""
        from prunestack.data.loader import load_dataset

        X = pd.DataFrame({"col": ["B", "A", "C", "A"] * 5})
        y = pd.Series([0, 1, 0, 1] * 5, name="target")

        ds1 = _make_cat_dataset(tmp_path / "run1", X, y)
        ds2 = _make_cat_dataset(tmp_path / "run2", X, y)

        b1 = load_dataset(ds1)
        b2 = load_dataset(ds2)

        np.testing.assert_array_equal(b1.X["col"].values, b2.X["col"].values)

    def test_string_labels_and_string_features(self, tmp_path):
        """Both string features AND string labels should be handled."""
        from prunestack.data.loader import load_dataset

        X = pd.DataFrame({"gender": ["Male", "Female"] * 10})
        y = pd.Series(["Yes", "No"] * 10, name="target")
        ds = _make_cat_dataset(tmp_path, X, y)
        bundle = load_dataset(ds)

        assert "gender" in bundle.feature_encoders
        assert bundle.label_encoder is not None
        assert bundle.X["gender"].dtype in [np.int64, int]
        assert bundle.y.dtype in [np.int64, np.int32, int]


class TestValidationCategoricalHandling:
    """Tests for validation.py handling of categorical features."""

    def test_validate_with_string_columns(self):
        """validate_dataset should not crash on string feature columns."""
        from prunestack.data.validation import validate_dataset

        X = pd.DataFrame({
            "gender": ["Male", "Female", "Male"],
            "age": [25.0, 30.0, 35.0],
        })
        y = np.array([0, 1, 0])
        result = validate_dataset(X, y)
        # Should succeed, possibly with a warning
        assert result.is_valid

    def test_validate_all_string_columns(self):
        """All-string feature matrix should not crash validation."""
        from prunestack.data.validation import validate_dataset

        X = pd.DataFrame({
            "a": ["x", "y", "z"],
            "b": ["p", "q", "r"],
        })
        y = np.array([0, 1, 0])
        result = validate_dataset(X, y)
        assert result.is_valid

    def test_validate_encoded_dataframe(self):
        """Already-encoded (all numeric) DataFrame should work as before."""
        from prunestack.data.validation import validate_dataset

        X = pd.DataFrame({
            "f1": [1.0, 2.0, 3.0],
            "f2": [4.0, 5.0, 6.0],
        })
        y = np.array([0, 1, 0])
        result = validate_dataset(X, y)
        assert result.is_valid
        # No warning about categorical columns
        assert not any("categorical" in w.lower() for w in result.warnings)

    def test_validate_numpy_string_array(self):
        """Validation should handle numpy array with string values."""
        from prunestack.data.validation import validate_dataset

        X = np.array([["Male", "Urban"], ["Female", "Rural"], ["Male", "Urban"]])
        y = np.array([0, 1, 0])
        result = validate_dataset(X, y)
        assert result.is_valid

    def test_validate_numpy_object_array(self):
        """Validation should handle numpy object-dtype array with mixed values."""
        from prunestack.data.validation import validate_dataset

        X = np.array(
            [["Male", 1], ["Female", 2], ["Male", 3]],
            dtype=object,
        )
        y = np.array([0, 1, 0])
        result = validate_dataset(X, y)
        assert result.is_valid

    def test_validate_numpy_unicode_dtype_array(self):
        """Validation should handle numpy '<U' unicode dtype arrays."""
        from prunestack.data.validation import validate_dataset

        X = np.array([["a", "b"], ["c", "d"], ["a", "b"]], dtype="<U10")
        y = np.array([0, 1, 0])
        result = validate_dataset(X, y)
        assert result.is_valid

    def test_validate_numpy_numeric_array_no_warning(self):
        """Pure numeric numpy array should not raise categorical warnings."""
        from prunestack.data.validation import validate_dataset

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0, 1, 0])
        result = validate_dataset(X, y)
        assert result.is_valid
        assert not any("categorical" in w.lower() for w in result.warnings)


class TestAllCategoricalDataset:
    """End-to-end loading of a dataset whose features and labels are all strings."""

    @pytest.fixture
    def bundle(self, tmp_path):
        from prunestack.data.loader import load_dataset

        rng = np.random.RandomState(0)
        n = 90
        X = pd.DataFrame({
            "Gender": rng.choice(["Male", "Female"], n),
            "Smoker": rng.choice(["yes", "no"], n),
            "Region": rng.choice(["north", "south", "east"], n),
        })
        y = pd.Series(rng.choice(["Yes", "No"], n), name="target")
        return load_dataset(_make_cat_dataset(tmp_path, X, y))

    def test_features_encoded(self, bundle):
        assert bundle.n_features == 3
        assert len(bundle.feature_encoders) == 3
        assert bundle.X.dtypes.apply(lambda dt: np.issubdtype(dt, np.integer)).all()

    def test_labels_encoded(self, bundle):
        assert set(bundle.label_encoder.classes_) == {"No", "Yes"}
        assert np.issubdtype(np.asarray(bundle.y).dtype, np.integer)

    def test_validation_passes(self, bundle):
        from prunestack.data.validation import validate_dataset

        assert validate_dataset(bundle.X, bundle.y).is_valid

    def test_feature_inverse_transform(self, bundle):
        encoder = bundle.feature_encoders["Gender"]
        assert set(encoder.inverse_transform(bundle.X["Gender"].values)) == {"Male", "Female"}
