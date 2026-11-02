"""Tests for prunestack.ensemble.stacking."""

from __future__ import annotations

import numpy as np
import pytest

from prunestack.ensemble.stacking import (
    META_LEARNERS,
    NNLSMultiResponseClassifier,
    NNLSRegressor,
    StackedEnsemble,
    available_meta_learners,
    build_meta_learner,
    one_hot_labels,
)


@pytest.fixture
def classification_data():
    """Three-class labels with ten noisy base models (20 % random errors)."""
    rng = np.random.RandomState(42)
    n_samples, n_models = 210, 10
    y = np.repeat([0, 1, 2], n_samples // 3)
    predictions = np.tile(y, (n_models, 1)).T
    noise = rng.random((n_samples, n_models)) < 0.2
    predictions[noise] = rng.randint(0, 3, noise.sum())
    return predictions.astype(float), y


@pytest.fixture
def regression_data():
    rng = np.random.RandomState(42)
    n_samples, n_models = 200, 10
    y = rng.normal(0, 50, n_samples)
    predictions = np.tile(y, (n_models, 1)).T + rng.normal(0, 10, (n_samples, n_models))
    return predictions, y


class TestNNLS:
    def test_regressor_weights_non_negative(self, regression_data):
        predictions, y = regression_data
        model = NNLSRegressor().fit(predictions, y)
        assert (model.weights_ >= 0).all()

    def test_regressor_prefers_exact_model(self, regression_data):
        predictions, y = regression_data
        predictions = predictions.copy()
        predictions[:, 3] = y
        model = NNLSRegressor().fit(predictions, y)
        assert np.argmax(model.weights_) == 3
        np.testing.assert_allclose(model.predict(predictions), y, atol=1e-6)

    def test_regressor_ignores_anti_correlated_model(self, regression_data):
        predictions, y = regression_data
        predictions = np.column_stack([predictions[:, 0], -y])
        model = NNLSRegressor().fit(predictions, y)
        assert model.weights_[1] == 0.0

    def test_regressor_falls_back_to_equal_weights(self):
        predictions = -np.abs(np.random.RandomState(0).rand(20, 4)) - 1
        y = np.abs(np.random.RandomState(1).rand(20)) + 1
        model = NNLSRegressor().fit(predictions, y)
        np.testing.assert_allclose(model.weights_, 0.25)

    def test_mlr_weights_shape_and_sign(self, classification_data):
        predictions, y = classification_data
        model = NNLSMultiResponseClassifier(n_classes=3).fit(predictions, y)
        assert model.weights_.shape == (3, 10)
        assert (model.weights_ >= 0).all()

    def test_mlr_prefers_exact_model(self, classification_data):
        predictions, y = classification_data
        predictions = predictions.copy()
        predictions[:, 7] = y
        model = NNLSMultiResponseClassifier(n_classes=3).fit(predictions, y)
        assert (np.argmax(model.weights_, axis=1) == 7).all()
        np.testing.assert_array_equal(model.predict(predictions), y)

    def test_mlr_equal_weights_is_majority_vote(self):
        predictions = np.array([[0, 0, 1], [1, 1, 0], [2, 2, 2]])
        model = NNLSMultiResponseClassifier(n_classes=3)
        model.weights_ = np.full((3, 3), 1 / 3)
        np.testing.assert_array_equal(model.predict(predictions), [0, 1, 2])

    def test_one_hot_labels(self):
        encoded = one_hot_labels(np.array([[0, 2], [1, 1]]), n_classes=3)
        np.testing.assert_array_equal(encoded, [[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0]])


class TestBuildMetaLearner:
    def test_unknown_name(self):
        with pytest.raises(ValueError, match="Unknown meta-learner"):
            build_meta_learner("lasso", "regression")

    def test_wrong_task(self):
        with pytest.raises(ValueError, match="does not support"):
            build_meta_learner("nnls_mlr", "regression")

    def test_available_meta_learners(self):
        assert available_meta_learners("classification")[0] == "nnls_mlr"
        assert "nnls" in available_meta_learners("regression")
        assert "logistic" not in available_meta_learners("regression")


class TestStackedEnsemble:
    def test_defaults(self, classification_data, regression_data):
        assert StackedEnsemble(*classification_data, "classification").meta_learner == "nnls_mlr"
        assert StackedEnsemble(*regression_data, "regression").meta_learner == "nnls"

    def test_invalid_task_type(self, regression_data):
        with pytest.raises(ValueError, match="task_type"):
            StackedEnsemble(*regression_data, "ranking")

    def test_invalid_meta_learner_fails_fast(self, regression_data):
        with pytest.raises(ValueError):
            StackedEnsemble(*regression_data, "regression", meta_learner="logistic")

    def test_cv_folds_capped_by_smallest_class(self):
        y = np.array([0] * 20 + [1] * 3)
        stacker = StackedEnsemble(np.column_stack([y, y]), y, "classification", cv_folds=5)
        assert stacker.cv_folds == 3

    def test_predict_before_train_raises(self, classification_data):
        with pytest.raises(RuntimeError, match="train"):
            StackedEnsemble(*classification_data, "classification").predict()

    def test_train_and_predict(self, classification_data):
        predictions, y = classification_data
        stacker = StackedEnsemble(predictions, y, "classification", cv_folds=3)
        stacker.train()
        assert stacker.is_trained
        assert stacker.predict().shape == (len(y),)
        assert stacker.predict([0, 1, 2]).shape == (len(y),)

    def test_stacking_beats_single_model(self, classification_data):
        predictions, y = classification_data
        stacker = StackedEnsemble(predictions, y, "classification", cv_folds=3)
        stacker.train()
        assert stacker.evaluate() > stacker.evaluate([0])

    def test_regression_rmse(self, regression_data):
        predictions, y = regression_data
        stacker = StackedEnsemble(predictions, y, "regression", cv_folds=3)
        stacker.train()
        single_rmse = float(np.sqrt(np.mean((predictions[:, 0] - y) ** 2)))
        assert 0 < stacker.evaluate() < single_rmse

    def test_deterministic(self, classification_data):
        runs = []
        for _ in range(2):
            stacker = StackedEnsemble(*classification_data, "classification", cv_folds=3, seed=7)
            stacker.train()
            runs.append(stacker.oof_meta_predictions)
        np.testing.assert_array_equal(*runs)

    @pytest.mark.parametrize("name", sorted(META_LEARNERS))
    def test_every_meta_learner_runs(self, name, classification_data, regression_data):
        for task_type in META_LEARNERS[name]:
            data = classification_data if task_type == "classification" else regression_data
            stacker = StackedEnsemble(*data, task_type, cv_folds=3, meta_learner=name)
            stacker.train()
            assert np.isfinite(stacker.evaluate())


class TestProbabilityInputs:
    """Stacking on out-of-fold class probabilities (Ting & Witten, 1999)."""

    @staticmethod
    def _setup(n=200, n_models=5, n_classes=3, seed=0):
        rng = np.random.RandomState(seed)
        y = rng.randint(0, n_classes, n)
        logits = rng.normal(0, 1, (n, n_models, n_classes))
        logits[np.arange(n), :, y] += np.linspace(0.5, 2.5, n_models)
        proba = np.exp(logits) / np.exp(logits).sum(axis=2, keepdims=True)
        labels = proba.argmax(axis=2).astype(float)
        return labels, proba, y

    def test_uses_probabilities(self):
        from prunestack.ensemble.stacking import StackedEnsemble
        labels, proba, y = self._setup()
        st = StackedEnsemble(labels, y, "classification", base_probabilities=proba)
        assert st.uses_probabilities and st.input_kind == "probabilities"
        st.train()
        assert 0.0 <= st.evaluate() <= 1.0
        assert st.predict([0, 4]).shape == (200,)

    def test_nnls_mlr_weights_on_probabilities(self):
        from prunestack.ensemble.stacking import NNLSMultiResponseClassifier
        labels, proba, y = self._setup()
        m = NNLSMultiResponseClassifier(3).fit(proba, y)
        assert m.weights_.shape == (3, 5) and (m.weights_ >= 0).all()
        # The most informative model (last) gets the largest weight in every class.
        assert (m.weights_.argmax(axis=1) == 4).all()

    def test_categorical_nb_falls_back_to_labels(self):
        from prunestack.ensemble.stacking import StackedEnsemble
        labels, proba, y = self._setup()
        st = StackedEnsemble(labels, y, "classification", meta_learner="categorical_nb",
                             base_probabilities=proba)
        assert not st.uses_probabilities and st.input_kind == "labels"

    @pytest.mark.parametrize("name", ["ridge", "logistic", "random_forest", "hist_gradient_boosting"])
    def test_sklearn_meta_learners_accept_probabilities(self, name):
        from prunestack.ensemble.stacking import StackedEnsemble
        labels, proba, y = self._setup(n=120)
        st = StackedEnsemble(labels, y, "classification", meta_learner=name, base_probabilities=proba)
        assert 0.0 <= st.evaluate([1, 2, 3]) <= 1.0

    def test_wrong_shape_raises(self):
        from prunestack.ensemble.stacking import StackedEnsemble
        labels, proba, y = self._setup()
        with pytest.raises(ValueError, match="base_probabilities"):
            StackedEnsemble(labels, y, "classification", base_probabilities=proba[:, :2])
