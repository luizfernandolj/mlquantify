import numpy as np
import pandas as pd
import pytest

from mlquantify import config_context
from mlquantify.base import BaseQuantifier
from mlquantify.multiclass import define_binary
from mlquantify.utils._tags import Tags, TargetInputTags, PredictionRequirements
from mlquantify.utils._validation import validate_data, validate_predictions, validate_prevalences
from mlquantify.utils.prevalence import get_prev_from_labels
from mlquantify.utils._sampling import get_indexes_with_prevalence


class DummyQuantifier(BaseQuantifier):
    def __init__(self, estimator_type="soft"):
        self._estimator_type = estimator_type

    def __mlquantify_tags__(self):
        return Tags(
            estimation_type=None,
            estimator_function=None,
            estimator_type=self._estimator_type,
            aggregation_type=None,
            target_input_tags=TargetInputTags(),
            prediction_requirements=PredictionRequirements(),
            has_estimator=False,
            requires_fit=True,
        )


@define_binary
class DummyBinaryQuantifier(BaseQuantifier):
    def __init__(self, constant=0.6, strategy="ovr", n_jobs=None):
        self.constant = constant
        self.strategy = strategy
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array([1.0 - self.constant, self.constant])

    def aggregate(self, preds, y_train):
        return np.array([1.0 - self.constant, self.constant])


def test_validate_data_accepts_numpy_and_pandas():
    q = DummyQuantifier()
    X_np = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    y_np = np.array([0, 1, 0])
    X_out, y_out = validate_data(q, X_np, y_np)
    assert isinstance(X_out, np.ndarray)
    assert isinstance(y_out, np.ndarray)

    X_pd = pd.DataFrame({"f1": [0.0, 1.0, 0.5], "f2": [1.0, 0.0, 0.5]})
    y_pd = pd.Series([0, 1, 0])
    X_out, y_out = validate_data(q, X_pd, y_pd)
    assert isinstance(X_out, np.ndarray)
    assert isinstance(y_out, np.ndarray)


def test_validate_data_accepts_categorical_features():
    q = DummyQuantifier()
    X_pd = pd.DataFrame({
        "cat": pd.Series(["a", "b", "a"], dtype="category"),
        "num": [1.0, 2.0, 3.0],
    })
    y_pd = pd.Series([0, 1, 0])
    X_out, y_out = validate_data(q, X_pd, y_pd)
    assert isinstance(X_out, np.ndarray)
    assert isinstance(y_out, np.ndarray)


def test_validate_data_rejects_nan_features():
    q = DummyQuantifier()
    X_pd = pd.DataFrame({"f1": [0.0, np.nan, 0.5], "f2": [1.0, 0.0, 0.5]})
    y_pd = pd.Series([0, 1, 0])
    with pytest.raises(ValueError):
        validate_data(q, X_pd, y_pd)


def test_validate_data_rejects_nan_labels():
    q = DummyQuantifier()
    X_np = np.array([[0.0], [1.0], [0.5]])
    y_pd = pd.Series([0.0, np.nan, 1.0])
    with pytest.raises(ValueError):
        validate_data(q, X_np, y_pd)


def test_get_prev_from_labels_string_labels():
    y = np.array(["a", "a", "b", "b", "b"])
    prev = get_prev_from_labels(y, format="dict", classes=["a", "b"])
    assert prev["a"] == pytest.approx(0.4)
    assert prev["b"] == pytest.approx(0.6)


def test_get_prev_from_labels_float_labels():
    y = np.array([0.1, 0.1, 0.2, 0.2, 0.2])
    prev = get_prev_from_labels(y, format="dict", classes=[0.1, 0.2])
    assert prev[0.1] == pytest.approx(0.4)
    assert prev[0.2] == pytest.approx(0.6)


def test_get_prev_from_labels_categorical_series():
    y = pd.Series(pd.Categorical(["x", "y", "x", "x"]))
    prev = get_prev_from_labels(y, format="dict", classes=["x", "y"])
    assert prev["x"] == pytest.approx(0.75)
    assert prev["y"] == pytest.approx(0.25)


def test_validate_prevalences_string_classes_sum_normalize():
    q = DummyQuantifier()
    classes = np.array(["a", "b", "c"])
    prev = {"a": 2.0, "b": 1.0}
    result = validate_prevalences(q, prev, classes, return_type="dict", normalize=True)
    assert set(result.keys()) == {"a", "b", "c"}
    assert pytest.approx(sum(result.values())) == 1.0


def test_validate_predictions_crisp_string_classes_2d():
    q = DummyQuantifier(estimator_type="crisp")
    preds = np.array([[0.2, 0.8], [0.7, 0.3]], dtype=float)
    y_train = np.array(["neg", "pos", "neg", "pos"])
    out = validate_predictions(q, preds, threshold=0.5, y_train=y_train)
    np.testing.assert_array_equal(out, np.array(["pos", "neg"]))


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_define_binary_string_labels(strategy):
    X = np.array([[0.0], [1.0], [0.5], [0.2], [0.8], [0.3]])
    y = np.array(["a", "b", "c", "a", "b", "c"])
    q = DummyBinaryQuantifier(strategy=strategy)
    q.fit(X, y)
    with config_context(prevalence_normalization="sum"):
        prev = q.predict(X)
    assert isinstance(prev, dict)
    assert set(prev.keys()) == {"a", "b", "c"}
    assert pytest.approx(sum(prev.values())) == 1.0


def test_get_indexes_with_prevalence_string_labels():
    y = np.array(["a"] * 50 + ["b"] * 50)
    prevalence = [0.7, 0.3]
    idx = get_indexes_with_prevalence(y, prevalence, 10, random_state=42)
    assert len(idx) == 10
    assert np.all((np.array(idx) >= 0) & (np.array(idx) < len(y)))
