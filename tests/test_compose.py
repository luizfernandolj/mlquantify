import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.base_aggregative import (
    AggregationMixin,
    SoftLearnerQMixin,
    is_aggregative_quantifier,
    uses_crisp_predictions,
    uses_soft_predictions,
)
from mlquantify.compose import (
    ComposeQuantifier,
    LeastSquaresLoss,
    LikelihoodComposeQuantifier,
    LinearComposeQuantifier,
)
from mlquantify.counting import AC
from mlquantify.matching import GHDx
from mlquantify.representations import HistogramRepresentation, PredictionRepresentation


class PredictOnlyLearner(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
        self.threshold_ = np.median(X[:, 0])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.where(X[:, 0] >= self.threshold_, self.classes_[-1], self.classes_[0])


class SoftLinearCompose(SoftLearnerQMixin, AggregationMixin, LinearComposeQuantifier):
    def __init__(
        self,
        representation,
        learner=None,
        loss="hellinger",
        solver="slsqp",
        normalize=None,
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            representation=representation,
            learner=learner,
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=normalize,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


class SoftLikelihoodCompose(SoftLearnerQMixin, AggregationMixin, LikelihoodComposeQuantifier):
    def __init__(
        self,
        learner=None,
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            representation=PredictionRepresentation(method="soft", average=False),
            learner=learner,
            solver=solver,
            aggregative=True,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


def test_compose_quantifier_aliases_linear_compose():
    assert ComposeQuantifier is LinearComposeQuantifier
    assert LeastSquaresLoss()([1.0, 2.0], [2.0, 4.0]) == pytest.approx(5.0)


def test_compose_mixins_define_learner_prediction_modes():
    linear = LinearComposeQuantifier(
        representation=PredictionRepresentation(method="soft", average=True),
        aggregative=False,
    )
    soft_linear = SoftLinearCompose(
        learner=LogisticRegression(),
        representation=PredictionRepresentation(method="soft", average=True),
    )
    likelihood = SoftLikelihoodCompose(learner=LogisticRegression())
    crisp = AC(learner=PredictOnlyLearner())
    non_aggregative = GHDx()

    assert not uses_soft_predictions(linear)
    assert not is_aggregative_quantifier(linear)
    assert uses_soft_predictions(soft_linear)
    assert is_aggregative_quantifier(soft_linear)
    assert uses_soft_predictions(likelihood)
    assert uses_crisp_predictions(crisp)
    assert not is_aggregative_quantifier(non_aggregative)


def test_generic_compose_requires_method_mixins_when_aggregative(binary_dataset):
    X, y = binary_dataset
    q = LinearComposeQuantifier(
        learner=LogisticRegression(max_iter=1000, random_state=42),
        representation=PredictionRepresentation(method="soft", average=True),
    )

    with pytest.raises(ValueError, match="AggregationMixin"):
        q.fit(X, y)


def test_aggregative_compose_requires_prediction_representation(binary_dataset):
    X, y = binary_dataset
    q = SoftLinearCompose(
        learner=LogisticRegression(max_iter=1000, random_state=42),
        representation=HistogramRepresentation(bins=[4], mode="histogram"),
    )

    with pytest.raises(ValueError, match="PredictionRepresentation"):
        q.fit(X, y)


def test_crisp_compose_method_accepts_predict_only_learner(binary_dataset):
    X, y = binary_dataset
    q = AC(learner=PredictOnlyLearner(), cv=3)

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


def test_linear_compose_aggregative_soft_representation(binary_dataset):
    X, y = binary_dataset
    q = SoftLinearCompose(
        learner=LogisticRegression(max_iter=1000, random_state=42),
        representation=PredictionRepresentation(method="soft", average=True),
        loss="ls",
        cv=3,
        solver="slsqp",
    )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)
    assert q.best_distance_ is not None


def test_linear_compose_accepts_loss_instances(binary_dataset):
    X, y = binary_dataset
    q = SoftLinearCompose(
        learner=LogisticRegression(max_iter=1000, random_state=42),
        representation=PredictionRepresentation(method="hard", average=True),
        loss=LeastSquaresLoss(),
        cv=3,
    )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


def test_linear_compose_non_aggregative_histogram(multiclass_dataset):
    X, y = multiclass_dataset
    q = LinearComposeQuantifier(
        representation=HistogramRepresentation(bins=[4], mode="histogram"),
        loss="hellinger",
        aggregative=False,
        solver="slsqp",
    )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)


def test_likelihood_compose_over_adjusted_posteriors(binary_dataset):
    X, y = binary_dataset
    q = SoftLikelihoodCompose(
        learner=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        solver="slsqp",
    )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)
    assert q.best_distance_ is not None
