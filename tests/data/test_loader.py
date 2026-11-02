"""Tests for dataset loader."""

import pytest
import numpy as np

from prunestack.data.loader import load_dataset


class TestLoadDataset:
    def test_tiny_dataset_loads(self, tiny_dataset_dir):
        bundle = load_dataset(tiny_dataset_dir, n_folds=3, seed=42)
        assert bundle.X.shape == (20, 5)
        assert bundle.y.shape == (20,)
        assert bundle.task_type == "classification"
        assert bundle.n_features == 5
        assert bundle.n_samples == 20
        assert len(bundle.folds) == 3

    def test_tiny_dataset_folds_no_overlap(self, tiny_dataset_dir):
        bundle = load_dataset(tiny_dataset_dir, n_folds=5, seed=42)
        all_val = []
        for train_idx, val_idx in bundle.folds:
            # No overlap between train and val
            assert len(set(train_idx) & set(val_idx)) == 0
            all_val.extend(val_idx.tolist())
        # Every sample appears in exactly one val fold
        assert sorted(all_val) == list(range(20))

    def test_tiny_dataset_stratified(self, tiny_dataset_dir):
        """Classification should use stratified K-Fold."""
        bundle = load_dataset(tiny_dataset_dir, n_folds=5, seed=42)
        for train_idx, val_idx in bundle.folds:
            # Check approximate stratification
            train_ratio = bundle.y.iloc[train_idx].mean()
            val_ratio = bundle.y.iloc[val_idx].mean()
            assert abs(train_ratio - val_ratio) < 0.3

    def test_tiny_dataset_reproducible(self, tiny_dataset_dir):
        b1 = load_dataset(tiny_dataset_dir, seed=42)
        b2 = load_dataset(tiny_dataset_dir, seed=42)
        for (t1, v1), (t2, v2) in zip(b1.folds, b2.folds):
            np.testing.assert_array_equal(t1, t2)
            np.testing.assert_array_equal(v1, v2)

    def test_different_seed_different_splits(self, tiny_dataset_dir):
        b1 = load_dataset(tiny_dataset_dir, seed=42)
        b2 = load_dataset(tiny_dataset_dir, seed=99)
        # At least one fold should differ
        differs = False
        for (t1, _), (t2, _) in zip(b1.folds, b2.folds):
            if not np.array_equal(t1, t2):
                differs = True
                break
        assert differs

    def test_classification_dataset(self, classification_dataset_dir):
        bundle = load_dataset(classification_dataset_dir, seed=42)
        assert bundle.task_type == "classification"
        assert bundle.n_features == 6
        assert bundle.n_samples == 120
        assert len(bundle.folds) == 5
        assert len(bundle.feature_names) == 6

    def test_regression_dataset(self, regression_dataset_dir):
        bundle = load_dataset(regression_dataset_dir, seed=42)
        assert bundle.task_type == "regression"
        assert bundle.n_features == 6
        assert len(bundle.folds) == 5

    def test_regression_folds_partition_samples(self, regression_dataset_dir):
        """Regression uses plain K-fold; validation folds partition the samples."""
        bundle = load_dataset(regression_dataset_dir, n_folds=5, seed=42)
        all_val = []
        for train_idx, val_idx in bundle.folds:
            assert len(set(train_idx) & set(val_idx)) == 0
            all_val.extend(val_idx.tolist())
        assert sorted(all_val) == list(range(bundle.n_samples))

    def test_missing_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Missing required"):
            load_dataset(tmp_path)

    def test_feature_names_preserved(self, tiny_dataset_dir):
        bundle = load_dataset(tiny_dataset_dir)
        assert bundle.feature_names == ["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"]


class TestUninformativeColumns:
    def _write(self, tmp_path, X):
        import pandas as pd
        X.to_csv(tmp_path / "X_test.csv")
        pd.DataFrame({"target": [i % 2 for i in range(len(X))]}).to_csv(tmp_path / "y_test.csv")
        pd.DataFrame({"f1": [0.5]}, index=["DT"]).to_csv(tmp_path / "metrics.csv")
        return tmp_path

    def _frame(self):
        import pandas as pd
        n = 30
        return pd.DataFrame({
            "num": np.arange(n, dtype=float), "const": 1.0,
            "id": [f"row{i}" for i in range(n)], "cat": ["a", "b", "c"] * 10,
        })

    def test_find_uninformative_columns(self):
        from prunestack.data.loader import find_uninformative_columns
        assert find_uninformative_columns(self._frame()) == {"const": "constant", "id": "identifier"}

    def test_dropped_on_load_and_user_columns(self, tmp_path):
        bundle = load_dataset(self._write(tmp_path, self._frame()), drop_columns=["cat"])
        assert bundle.feature_names == ["num"]
        assert bundle.dropped_columns == {"cat": "user", "const": "constant", "id": "identifier"}

    def test_unknown_drop_column_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            load_dataset(self._write(tmp_path, self._frame()), drop_columns=["missing"])
