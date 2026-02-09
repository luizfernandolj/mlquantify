import numpy as np
import pytest

from mlquantify.base import BaseQuantifier
from mlquantify.utils._constraints import Interval, Options, CallableConstraint
from mlquantify.utils._exceptions import InputValidationError, InvalidParameterError, NotFittedError
from mlquantify.utils._tags import Tags, TargetInputTags, PredictionRequirements
from mlquantify.utils._validation import (
    validate_y,
    validate_predictions,
    validate_prevalences,
    normalize_prevalences,
    check_is_fitted,
    check_classes_attribute,
    validate_parameter_constraints,
)


class DummyQuantifier(BaseQuantifier):
    def __init__(self, target_tags=None, estimator_type="soft"):
        self._target_tags = target_tags or TargetInputTags()
        self._estimator_type = estimator_type

    def __mlquantify_tags__(self):
        return Tags(
            estimation_type=None,
            estimator_function=None,
            estimator_type=self._estimator_type,
            aggregation_type=None,
            target_input_tags=self._target_tags,
            prediction_requirements=PredictionRequirements(),
            has_estimator=False,
            requires_fit=True,
        )

    def fit(self, X=None, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.array([0.5, 0.5])


VALIDATE_Y_CASES = [
    (TargetInputTags(one_d=True, two_d=False, multi_class=False, categorical=True), np.array([0, 1, 0, 1]), True),
    (TargetInputTags(one_d=True, two_d=False, multi_class=False, categorical=True), np.array([0, 1, 2]), False),
    (TargetInputTags(one_d=False, two_d=True, multi_class=True), np.array([[0.5, 0.5], [0.2, 0.8]]), True),
    (TargetInputTags(one_d=False, two_d=True, multi_class=True), np.array([[0.5, 0.6], [0.2, 0.8]]), False),
    (TargetInputTags(one_d=True, two_d=False, multi_class=True, categorical=False), np.array(["a", "b"]), False),
    (TargetInputTags(one_d=True, two_d=False, continuous=True), np.array([0.1, 0.2, 0.3]), True),
    (TargetInputTags(one_d=False, two_d=True, multi_class=True), np.array([0, 1, 0]), False),
]


@pytest.mark.parametrize("target_tags, y, expected", VALIDATE_Y_CASES)
def test_validate_y(target_tags, y, expected):
    q = DummyQuantifier(target_tags=target_tags, estimator_type="soft")
    if expected:
        validate_y(q, y)
    else:
        with pytest.raises(InputValidationError):
            validate_y(q, y)


def test_validate_predictions_soft_rejects_int():
    q = DummyQuantifier(estimator_type="soft")
    preds = np.array([0, 1, 0], dtype=int)
    with pytest.raises(InputValidationError):
        validate_predictions(q, preds)


def test_validate_predictions_crisp_converts_1d():
    q = DummyQuantifier(estimator_type="crisp")
    preds = np.array([0.2, 0.8, 0.1], dtype=float)
    out = validate_predictions(q, preds, threshold=0.5)
    np.testing.assert_array_equal(out, np.array([0, 1, 0]))


def test_validate_predictions_crisp_binary_with_classes():
    q = DummyQuantifier(estimator_type="crisp")
    preds = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=float)
    y_train = np.array([1, 2, 1, 2])
    out = validate_predictions(q, preds, threshold=0.5, y_train=y_train)
    np.testing.assert_array_equal(out, np.array([2, 1]))


def test_validate_predictions_crisp_multiclass():
    q = DummyQuantifier(estimator_type="crisp")
    preds = np.array([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]], dtype=float)
    y_train = np.array([0, 1, 2, 2])
    out = validate_predictions(q, preds, y_train=y_train)
    np.testing.assert_array_equal(out, np.array([2, 0]))


def test_validate_prevalences_returns_dict_when_requested():
    q = DummyQuantifier()
    classes = np.array([0, 1])
    prev = np.array([2.0, 1.0])
    result = validate_prevalences(q, prev, classes, return_type="dict", normalize=True)
    assert isinstance(result, dict)
    assert pytest.approx(sum(result.values())) == 1.0


def test_validate_prevalences_returns_array_when_requested():
    q = DummyQuantifier()
    classes = np.array([0, 1])
    prev = np.array([2.0, 1.0])
    result = validate_prevalences(q, prev, classes, return_type="array", normalize=True)
    assert isinstance(result, np.ndarray)
    assert pytest.approx(np.sum(result)) == 1.0


NORMALIZE_CASES = [
    (np.array([2.0, 1.0, 1.0]), np.array([0, 1, 2]), "sum"),
    (np.array([2.0, 1.0, 1.0]), np.array([0, 1, 2]), "l1"),
    (np.array([2.0, 1.0, 1.0]), np.array([0, 1, 2]), "softmax"),
    (np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([0, 1]), "mean"),
    (np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([0, 1]), "median"),
]


@pytest.mark.parametrize("prev, classes, method", NORMALIZE_CASES)
def test_normalize_prevalences_methods(prev, classes, method):
    result = normalize_prevalences(prev, classes, method=method)
    assert np.isfinite(np.asarray(result)).all()


def test_check_classes_attribute_passthrough():
    class Dummy:
        pass

    q = Dummy()
    classes = np.array([0, 1])
    out = check_classes_attribute(q, classes)
    np.testing.assert_array_equal(out, classes)


def test_check_classes_attribute_same():
    class Dummy:
        pass

    q = Dummy()
    q.classes_ = np.array([0, 1])
    out = check_classes_attribute(q, np.array([0, 1]))
    np.testing.assert_array_equal(out, q.classes_)


def test_check_is_fitted_raises():
    q = DummyQuantifier()
    with pytest.raises(NotFittedError):
        check_is_fitted(q)


def test_check_is_fitted_passes():
    q = DummyQuantifier().fit()
    check_is_fitted(q)


def test_validate_parameter_constraints_valid():
    constraints = {
        "alpha": [Interval(0.0, 1.0)],
        "mode": [Options(["a", "b"])],
        "fn": [CallableConstraint()],
        "arr": ["array-like"],
    }
    params = {
        "alpha": 0.5,
        "mode": "a",
        "fn": lambda x: x,
        "arr": [1, 2, 3],
    }
    validate_parameter_constraints(constraints, params, caller_name="Dummy")


def test_validate_parameter_constraints_invalid():
    constraints = {
        "alpha": [Interval(0.0, 1.0)],
        "mode": [Options(["a", "b"])],
    }
    params = {"alpha": 2.0, "mode": "c"}
    with pytest.raises(InvalidParameterError):
        validate_parameter_constraints(constraints, params, caller_name="Dummy")
