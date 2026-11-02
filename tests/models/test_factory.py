"""Tests for prunestack.models.factory."""

import pytest

from prunestack.models.factory import create_model_pool


@pytest.fixture
def classification_hyperparameters():
    return {
        "DecisionTreeClassifier": {"random_state": 42},
        "KNeighborsClassifier": {"n_neighbors": 5},
        "MLPClassifier": {"hidden_layer_sizes": (100,), "max_iter": 200, "random_state": 42},
        "LogisticRegression": {"max_iter": 1000},
    }


@pytest.fixture
def regression_hyperparameters():
    return {
        "DecisionTreeRegressor": {"random_state": 42},
        "KNeighborsRegressor": {"n_neighbors": 5},
        "MLPRegressor": {"max_iter": 200, "random_state": 42},
        "Ridge": {},
    }


def make_pool(hyperparameters, task_type="classification", n_features=20, **kwargs):
    return create_model_pool(
        n_features=n_features, hyperparameters=hyperparameters, task_type=task_type, **kwargs,
    )


class TestCreateModelPool:
    def test_default_pool_has_20_models_per_algorithm(self, classification_hyperparameters):
        assert len(make_pool(classification_hyperparameters)) == 4 * 20

    def test_custom_pool_size(self, classification_hyperparameters):
        assert len(make_pool(classification_hyperparameters, max_models_per_algo=5)) == 4 * 5

    def test_model_ids_sequential(self, classification_hyperparameters):
        pool = make_pool(classification_hyperparameters)
        assert [m.model_id for m in pool] == list(range(len(pool)))

    def test_one_original_and_19_variants(self, classification_hyperparameters):
        pool = make_pool(classification_hyperparameters)
        for algo in classification_hyperparameters:
            models = [m for m in pool if m.algorithm == algo]
            originals = [m for m in models if m.is_original]
            assert len(originals) == 1
            assert originals[0].n_features_used == 20
            assert originals[0].name == f"{algo}_original"
            assert len(models) - 1 == 19
            assert models[-1].name == f"{algo}_subset_19"

    def test_variants_use_60_to_90_percent_of_features(self, classification_hyperparameters):
        pool = make_pool(classification_hyperparameters)
        for m in pool:
            assert all(0 <= i < 20 for i in m.feature_indices)
            assert len(m.feature_indices) == len(set(m.feature_indices))
            if not m.is_original:
                assert 12 <= m.n_features_used <= 18

    def test_deterministic(self, classification_hyperparameters):
        first = make_pool(classification_hyperparameters)
        second = make_pool(classification_hyperparameters)
        assert [m.to_dict() for m in first] == [m.to_dict() for m in second]

    def test_regression_pool(self, regression_hyperparameters):
        assert len(make_pool(regression_hyperparameters, "regression", n_features=30)) == 80

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            make_pool({"Ridge": {}}, "classification")

    def test_to_dict_is_json_ready(self, classification_hyperparameters):
        pool = make_pool(classification_hyperparameters)
        assert {"model_id", "name", "algorithm", "feature_indices"} <= set(pool[0].to_dict())
        mlp = next(m for m in pool if m.algorithm == "MLPClassifier")
        assert isinstance(mlp.to_dict()["params"]["hidden_layer_sizes"], list)

    def test_params_are_copies(self, classification_hyperparameters):
        pool = make_pool(classification_hyperparameters)
        pool[0].params["mutated"] = True
        assert "mutated" not in pool[1].params
