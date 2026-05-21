import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.base import BaseQuantifier
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
from mlquantify.utils._get_scores import apply_cross_validation


class PredictOnlyLearner(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X = np.asarray(X)
        self.classes_ = np.unique(y)
        self.threshold_ = np.median(X[:, 0])
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.where(X[:, 0] >= self.threshold_, self.classes_[-1], self.classes_[0])


class TrackingProbabilisticLearner(BaseEstimator, ClassifierMixin):
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


class FixedProbabilityLearner:
    def __init__(self, classes, probabilities):
        self.classes_ = np.asarray(classes)
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        return self.probabilities[: len(X)]


class FixedPredictLearner:
    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)

    def predict(self, X):
        return self.predictions[: len(X)]


class FixedDecisionLearner:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def decision_function(self, X):
        return self.values[: len(X)]


class DecisionLearnerQMixin:
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator_function = "decision_function"
        tags.estimator_type = "soft"
        return tags


class DecisionAggregation(DecisionLearnerQMixin, AggregationMixin, BaseQuantifier):
    def __init__(self, learner=None):
        self.learner = learner


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


def test_apply_cross_validation_clones_and_refits_original():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    learner = TrackingProbabilisticLearner()
    TrackingProbabilisticLearner.fit_records = []

    predictions, labels, fitted_learner = apply_cross_validation(
        learner,
        X,
        y,
        cv=3,
        cv_prediction="refit",
    )

    fold_records = TrackingProbabilisticLearner.fit_records[:-1]
    final_record = TrackingProbabilisticLearner.fit_records[-1]

    assert predictions.shape == (12, 2)
    assert labels.shape == (12,)
    assert fitted_learner is learner
    assert len(fold_records) == 3
    assert len({record[0] for record in fold_records}) == 3
    assert all(record[0] != id(learner) for record in fold_records)
    assert final_record == (id(learner), len(y))


def test_apply_cross_validation_ensemble_returns_fold_estimators():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = np.asarray([0, 1] * 6)
    learner = TrackingProbabilisticLearner()
    TrackingProbabilisticLearner.fit_records = []

    _, _, fitted_learner = apply_cross_validation(
        learner,
        X,
        y,
        cv=3,
        cv_prediction="ensemble",
    )

    assert isinstance(fitted_learner, list)
    assert len(fitted_learner) == 3
    assert all(estimator is not learner for estimator in fitted_learner)
    assert len(TrackingProbabilisticLearner.fit_records) == 3
    assert not hasattr(learner, "classes_")


def test_aggregation_mixin_predicts_with_cv_ensemble(binary_dataset):
    X, y = binary_dataset
    q = SoftLinearCompose(
        learner=TrackingProbabilisticLearner(),
        representation=PredictionRepresentation(method="soft", average=True),
        loss="ls",
        cv=3,
    )

    q.fit(X, y, cv_prediction="ensemble")
    ensemble_prediction = q._predict_learner(X[:5])
    manual_prediction = np.mean(
        [estimator.predict_proba(X[:5]) for estimator in q.learner_],
        axis=0,
    )

    assert isinstance(q.learner_, list)
    assert ensemble_prediction == pytest.approx(manual_prediction)


def test_aggregation_mixin_aligns_ensemble_probabilities():
    X = np.zeros((2, 1))
    q = SoftLinearCompose(
        learner=FixedProbabilityLearner([0, 1], [[0.5, 0.5], [0.5, 0.5]]),
        representation=PredictionRepresentation(method="soft", average=True),
    )
    q.classes_ = np.asarray([0, 1])
    q.learner_ = [
        FixedProbabilityLearner([0, 1], [[0.8, 0.2], [0.1, 0.9]]),
        FixedProbabilityLearner([1, 0], [[0.6, 0.4], [0.7, 0.3]]),
    ]

    prediction = q._predict_learner(X)

    assert prediction == pytest.approx(
        np.asarray([[0.6, 0.4], [0.2, 0.8]])
    )


def test_aggregation_mixin_uses_majority_vote_for_predict_ensemble():
    X = np.zeros((4, 1))
    q = AC(learner=FixedPredictLearner([0, 0, 0, 0]))
    q.classes_ = np.asarray([0, 1])
    q.learner_ = [
        FixedPredictLearner([0, 1, 1, 1]),
        FixedPredictLearner([1, 1, 0, 1]),
        FixedPredictLearner([1, 0, 0, 1]),
    ]

    prediction = q._predict_learner(X)

    assert prediction.tolist() == [1, 1, 0, 1]


def test_aggregation_mixin_averages_decision_function_ensemble():
    X = np.zeros((3, 1))
    q = DecisionAggregation(learner=FixedDecisionLearner([0.0, 0.0, 0.0]))
    q.learner_ = [
        FixedDecisionLearner([1.0, 2.0, 3.0]),
        FixedDecisionLearner([3.0, 4.0, 5.0]),
    ]

    prediction = q._predict_learner(X)

    assert prediction == pytest.approx(np.asarray([2.0, 3.0, 4.0]))


def test_learner_fitted_ignores_cv_prediction(binary_dataset):
    X, y = binary_dataset
    learner = TrackingProbabilisticLearner().fit(X, y)
    q = SoftLinearCompose(
        learner=learner,
        representation=PredictionRepresentation(method="soft", average=True),
        loss="ls",
        cv=3,
    )

    q.fit(X, y, learner_fitted=True, cv_prediction="ensemble")

    assert q.learner_ is learner
