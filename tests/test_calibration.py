"""Tests for :mod:`mlquantify.calibration` (temperature / vector scaling)."""

import numpy as np
import pytest

from mlquantify.calibration import (
    Calibrator,
    ClassifierCalibrator,
    QuantifierCalibrator,
)
from mlquantify.calibration._base import _nll

METHODS = ["ts", "bcts", "vs", "nbvs"]


@pytest.fixture
def calib_data():
    """3-class logits, probabilities and labels sampled from those probabilities."""
    rng = np.random.default_rng(0)
    n, K = 4000, 3
    z = rng.normal(size=(n, K))
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=row) for row in p])
    return z, y, p


def test_classes_exist_and_subclass():
    assert issubclass(ClassifierCalibrator, Calibrator)
    assert issubclass(QuantifierCalibrator, Calibrator)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("input_type", ["proba", "logits"])
def test_predict_returns_valid_simplex(calib_data, method, input_type):
    z, y, p = calib_data
    src = p if input_type == "proba" else z
    out = ClassifierCalibrator(method, input_type=input_type).fit(y, src).predict(src)
    assert out.shape == p.shape
    assert np.allclose(out.sum(axis=1), 1.0)
    assert (out >= 0).all() and (out <= 1).all()


@pytest.mark.parametrize("method", METHODS)
def test_calibration_does_not_increase_nll(calib_data, method):
    # Each method includes the identity map in its feasible set, so the fitted
    # NLL can never exceed the uncalibrated NLL.
    z, y, p = calib_data
    onehot = np.eye(z.shape[1])[y]
    cal = ClassifierCalibrator(method, input_type="logits").fit(y, z)
    before = _nll(z, onehot)
    after = _nll(np.log(cal.predict(z)), onehot)
    assert after <= before + 1e-8


def test_temperature_scaling_recovers_temperature():
    rng = np.random.default_rng(1)
    n, K, scale = 8000, 3, 2.0
    z = rng.normal(size=(n, K))
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=row) for row in p])
    # Sharpen the logits by ``scale`` (over-confident); TS should undo it with a
    # shared weight close to 1 / scale.
    cal = ClassifierCalibrator("ts", input_type="logits").fit(y, z * scale)
    assert cal.weights_[0] == pytest.approx(1.0 / scale, abs=0.1)
    assert np.allclose(cal.weights_, cal.weights_[0])  # one shared temperature
    assert np.allclose(cal.biases_, 0.0)


def test_temperature_scaling_preserves_argmax(calib_data):
    # A positive temperature rescales logits without changing their ordering.
    z, y, p = calib_data
    out = ClassifierCalibrator("ts", input_type="logits").fit(y, z).predict(z)
    assert np.array_equal(out.argmax(axis=1), z.argmax(axis=1))


def test_invalid_arguments():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=20)
    y = rng.integers(0, 3, size=20)
    with pytest.raises(ValueError):
        ClassifierCalibrator(method="bogus").fit(y, p)
    with pytest.raises(ValueError):
        ClassifierCalibrator(input_type="weird").fit(y, p)
    with pytest.raises(ValueError):
        ClassifierCalibrator().fit(y, p[:, 0])  # 1-D y_pred


def test_quantifier_calibrator_is_deferred():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=10)
    y = rng.integers(0, 3, size=10)
    with pytest.raises(NotImplementedError):
        QuantifierCalibrator().fit(y, p)
    with pytest.raises(NotImplementedError):
        QuantifierCalibrator().predict(p)


@pytest.mark.parametrize("method", METHODS)
def test_emq_integration(method):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from mlquantify.likelihood import EMQ

    X, y = make_classification(n_samples=900, n_features=10, n_informative=6,
                               n_classes=3, random_state=0)
    q = EMQ(LogisticRegression(max_iter=1000), calib_function=method).fit(X[:650], y[:650])
    prev = np.asarray(q.predict(X[650:]))
    assert abs(prev.sum() - 1.0) < 1e-6 and (prev >= 0).all()
