import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import (
    AggregationMixin,
    AggregativeMixin,
    CrispPredictionMixin,
    SoftPredictionMixin,
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
from mlquantify.utils._get_scores import apply_cross_validation


class PredictOnlyEstimator(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
        self.threshold_ = np.median(X[:, 0])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.where(X[:, 0] >= self.threshold_, self.classes_[-1], self.classes_[0])


class TrackingProbabilisticEstimator(BaseEstimator, ClassifierMixin):
    fit_records = []

    def __init__(self, positive_probability=0.7):
        self.positive_probability = positive_probability

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.train_size_ = len(y)
        self.fit_records.append((id(self), self.train_size_))
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        probabilities = np.zeros((X.shape[0], len(self.classes_)))

        if len(self.classes_) == 1:
            probabilities[:, 0] = 1.0
            return probabilities

        probabilities[:, 0] = 1.0 - self.positive_probability
        probabilities[:, 1] = self.positive_probability
        return probabilities


class FixedProbabilityEstimator:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        return self.probabilities[: len(X)]


class FixedPredictEstimator:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, X):
        return self.predictions[: len(X)]


class FixedDecisionEstimator:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def decision_function(self, X):
        return self.values[: len(X)]


class DecisionEstimatorQMixin:
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator_function = "decision_function"
        tags.estimator_type = "soft"
        return tags


class DecisionAggregation(DecisionEstimatorQMixin, AggregativeMixin, BaseQuantifier):
    def __init__(self, estimator=None):
        self.estimator = estimator


class SoftLinearCompose(SoftPredictionMixin, AggregativeMixin, LinearComposeQuantifier):
    def __init__(
        self,
        representation,
        estimator=None,
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
            estimator=estimator,
            loss=loss,
            solver=solver,
            aggregative=True,
            normalize=normalize,
        )
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state


class SoftLikelihoodCompose(SoftPredictionMixin, AggregativeMixin, LikelihoodComposeQuantifier):
    def __init__(
        self,
        estimator=None,
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        super().__init__(
            representation=PredictionRepresentation(method="soft", average=False),
            estimator=estimator,
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


def test_compose_mixins_define_estimator_prediction_modes():
    linear = LinearComposeQuantifier(
        representation=PredictionRepresentation(method="soft", average=True),
        aggregative=False,
    )
    soft_linear = SoftLinearCompose(
        estimator=LogisticRegression(),
        representation=PredictionRepresentation(method="soft", average=True),
    )
    likelihood = SoftLikelihoodCompose(estimator=LogisticRegression())
    crisp = AC(estimator=PredictOnlyEstimator())
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
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        representation=PredictionRepresentation(method="soft", average=True),
    )

    with pytest.raises(ValueError, match="AggregativeMixin"):
        q.fit(X, y)


def test_aggregative_compose_requires_prediction_representation(binary_dataset):
    X, y = binary_dataset
    q = SoftLinearCompose(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        representation=HistogramRepresentation(bins=[4], mode="histogram"),
    )

    with pytest.raises(ValueError, match="PredictionRepresentation"):
        q.fit(X, y)


def test_crisp_compose_method_accepts_predict_only_estimator(binary_dataset):
    X, y = binary_dataset
    q = AC(estimator=PredictOnlyEstimator(), cv=3)

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


def test_aggregation_mixin_name_remains_compatibility_alias():
    assert AggregationMixin is AggregativeMixin


def test_aggregative_quantifier_exposes_estimator_as_canonical_param():
    estimator = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    q = AC(estimator=estimator, cv=3)

    params = q.get_params(deep=True)

    assert params["estimator"] is estimator
    assert params["estimator__C"] == pytest.approx(0.5)

    q.set_params(estimator__C=0.25)
    cloned = clone(q)

    assert estimator.C == pytest.approx(0.25)
    assert isinstance(cloned.estimator, LogisticRegression)
    assert cloned.estimator is not estimator
    assert cloned.estimator.C == pytest.approx(0.25)
    assert not hasattr(cloned, "classes_")


def test_estimator_maps_to_sklearn_params(binary_dataset):
    X, y = binary_dataset
    estimator = LogisticRegression(max_iter=1000, random_state=42)
    q = AC(estimator=estimator, cv=3)

    assert q.get_params(deep=False)["estimator"] is estimator

    q.fit(X, y)

    assert q.estimator_ is estimator


def test_linear_compose_aggregative_soft_representation(binary_dataset):
    X, y = binary_dataset
    q = SoftLinearCompose(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
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
        estimator=LogisticRegression(max_iter=1000, random_state=42),
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
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
        solver="slsqp",
    )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)
    assert q.best_distance_ is not None


def test_apply_cross_validation_clones_and_refits_original():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    base_estimator = TrackingProbabilisticEstimator()
    TrackingProbabilisticEstimator.fit_records = []

    predictions, labels, fitted_estimator = apply_cross_validation(
        base_estimator,
        X,
        y,
        cv=3,
        cv_prediction="refit",
    )

    fold_records = TrackingProbabilisticEstimator.fit_records[:-1]
    final_record = TrackingProbabilisticEstimator.fit_records[-1]

    assert predictions.shape == (12, 2)
    assert labels.shape == (12,)
    assert fitted_estimator is base_estimator
    assert len(fold_records) == 3
    assert len({record[0] for record in fold_records}) == 3
    assert all(record[0] != id(base_estimator) for record in fold_records)
    assert final_record == (id(base_estimator), len(y))


def test_apply_cross_validation_ensemble_returns_fold_estimators():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    base_estimator = TrackingProbabilisticEstimator()
    TrackingProbabilisticEstimator.fit_records = []

    _, _, fitted_estimator = apply_cross_validation(
        base_estimator,
        X,
        y,
        cv=3,
        cv_prediction="ensemble",
    )

    assert isinstance(fitted_estimator, list)
    assert len(fitted_estimator) == 3
    assert all(estimator is not base_estimator for estimator in fitted_estimator)
    assert len(TrackingProbabilisticEstimator.fit_records) == 3
    assert not hasattr(base_estimator, "classes_")


def test_aggregation_mixin_predicts_with_cv_ensemble(binary_dataset):
    X, y = binary_dataset
    q = SoftLinearCompose(
        estimator=TrackingProbabilisticEstimator(),
        representation=PredictionRepresentation(method="soft", average=True),
        loss="ls",
        cv=3,
    )

    q.fit(X, y, cv_prediction="ensemble")
    ensemble_prediction = q._predict_estimator(X[:5])
    manual_prediction = np.mean(
        [estimator.predict_proba(X[:5]) for estimator in q.estimator_],
        axis=0,
    )

    assert isinstance(q.estimator_, list)
    assert ensemble_prediction == pytest.approx(manual_prediction)


def test_aggregation_mixin_aligns_ensemble_probabilities():
    X = np.zeros((2, 1))
    q = SoftLinearCompose(
        estimator=FixedProbabilityEstimator([0, 1], [[0.5, 0.5], [0.5, 0.5]]),
        representation=PredictionRepresentation(method="soft", average=True),
    )
    q.classes_ = np.asarray([0, 1])
    q.estimator_ = [
        FixedProbabilityEstimator([0, 1], [[0.8, 0.2], [0.1, 0.9]]),
        FixedProbabilityEstimator([1, 0], [[0.6, 0.4], [0.7, 0.3]]),
    ]

    prediction = q._predict_estimator(X)

    assert prediction == pytest.approx(
        np.asarray([[0.6, 0.4], [0.2, 0.8]])
    )


def test_aggregation_mixin_uses_majority_vote_for_predict_ensemble():
    X = np.zeros((4, 1))
    q = AC(estimator=FixedPredictEstimator([0, 0, 0, 0]))
    q.classes_ = np.asarray([0, 1])
    q.estimator_ = [
        FixedPredictEstimator([0, 1, 1, 1]),
        FixedPredictEstimator([1, 1, 0, 1]),
        FixedPredictEstimator([1, 0, 0, 1]),
    ]

    prediction = q._predict_estimator(X)

    assert prediction.tolist() == [1, 1, 0, 1]


def test_aggregation_mixin_averages_decision_function_ensemble():
    X = np.zeros((3, 1))
    q = DecisionAggregation(estimator=FixedDecisionEstimator([0.0, 0.0, 0.0]))
    q.estimator_ = [
        FixedDecisionEstimator([1.0, 2.0, 3.0]),
        FixedDecisionEstimator([3.0, 4.0, 5.0]),
    ]

    prediction = q._predict_estimator(X)

    assert prediction == pytest.approx(np.asarray([2.0, 3.0, 4.0]))


def test_estimator_fitted_ignores_cv_prediction(binary_dataset):
    X, y = binary_dataset
    estimator = TrackingProbabilisticEstimator().fit(X, y)
    q = SoftLinearCompose(
        estimator=estimator,
        representation=PredictionRepresentation(method="soft", average=True),
        loss="ls",
        cv=3,
    )

    q.fit(X, y, estimator_fitted=True, cv_prediction="ensemble")

    assert q.estimator_ is estimator
