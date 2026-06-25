"""Tests for :mod:`mlquantify.calibration`.

The calibration classes (``Calibrator``, ``ClassifierCalibrator``,
``QuantifierCalibrator``) are currently **unimplemented stubs** -- every method
body is just ``pass`` -- even though the README advertises them as a feature.

These tests pin the *intended* contract and are marked ``xfail`` so that:

* they do not lock in the current empty behaviour, and
* they will flip to passing automatically once the methods are implemented.

Until then they serve as executable documentation of the gap.
"""

import numpy as np
import pytest

from mlquantify.calibration import (
    Calibrator,
    ClassifierCalibrator,
    QuantifierCalibrator,
)


def test_calibration_classes_exist():
    # The public classes are importable and share the Calibrator interface.
    assert issubclass(ClassifierCalibrator, Calibrator)
    assert issubclass(QuantifierCalibrator, Calibrator)
    for cls in (Calibrator, ClassifierCalibrator, QuantifierCalibrator):
        assert hasattr(cls, "fit") and hasattr(cls, "predict")


@pytest.mark.xfail(reason="ClassifierCalibrator is an unimplemented stub", strict=False)
def test_classifier_calibrator_returns_probabilities():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.random(size=200)

    cal = ClassifierCalibrator().fit(y_true, y_pred)
    calibrated = cal.predict(y_pred)

    assert calibrated is not None
    calibrated = np.asarray(calibrated)
    assert calibrated.shape == y_pred.shape
    assert np.all((calibrated >= 0) & (calibrated <= 1))


@pytest.mark.xfail(reason="QuantifierCalibrator is an unimplemented stub", strict=False)
def test_quantifier_calibrator_returns_prevalences():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.random(size=200)

    cal = QuantifierCalibrator().fit(y_true, y_pred)
    calibrated = cal.predict(y_pred)

    assert calibrated is not None
