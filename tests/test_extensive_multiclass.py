import numpy as np
import pytest

from mlquantify import config_context
from mlquantify.base import BaseQuantifier
from mlquantify.multiclass import define_binary


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


def _make_multiclass_data():
    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.3, 0.7],
        ]
    )
    y = np.array([0, 1, 2, 0, 1, 2])
    return X, y


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_define_binary_multiclass_predict(strategy):
    X, y = _make_multiclass_data()
    q = DummyBinaryQuantifier(strategy=strategy)
    q.fit(X, y)
    with config_context(prevalence_normalization="sum"):
        prev = q.predict(X)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values())) == 1.0
    assert set(prev.keys()) == set(np.unique(y))


@pytest.mark.parametrize("strategy", ["ovr", "ovo"])
def test_define_binary_multiclass_fit_predict(strategy):
    X, y = _make_multiclass_data()
    q = DummyBinaryQuantifier(strategy=strategy)
    with config_context(prevalence_normalization="sum"):
        prev = q.fit_predict(X, y, X)
    assert isinstance(prev, dict)
    assert pytest.approx(sum(prev.values())) == 1.0
    assert set(prev.keys()) == set(np.unique(y))


def test_define_binary_binary_passthrough():
    X = np.array([[0.0], [1.0], [0.5], [0.2]])
    y = np.array([0, 1, 0, 1])
    q = DummyBinaryQuantifier()
    q.fit(X, y)
    prev = q.predict(X)
    assert isinstance(prev, np.ndarray)
    assert prev.shape[0] == 2


def test_define_binary_invalid_strategy_raises():
    X, y = _make_multiclass_data()
    q = DummyBinaryQuantifier(strategy="bad")
    with pytest.raises(ValueError):
        q.fit(X, y)
