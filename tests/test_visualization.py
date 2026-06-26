"""Smoke + behaviour tests for mlquantify.visualization.

Uses the non-interactive Agg backend so figures never try to open a window.
matplotlib is the optional ``viz`` extra, so skip this whole module when it is
not installed (e.g. a plain ``pip install -e .`` without ``[viz]``).
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from mlquantify.counting import CC
from mlquantify.confidence import construct_confidence_region
from mlquantify.visualization import (
    BiasDisplay,
    ConfidenceRegionDisplay,
    DiagonalDisplay,
    ErrorByShiftDisplay,
    PrevalenceDisplay,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def protocol_predictions():
    """A small synthetic protocol run: (true, predicted) prevalence arrays."""
    rng = np.random.default_rng(0)
    true = rng.dirichlet([4, 4], size=40)
    pred = np.clip(true + rng.normal(0, 0.05, true.shape), 0, 1)
    pred /= pred.sum(axis=1, keepdims=True)
    return true, pred


@pytest.fixture
def protocol_predictions_3():
    rng = np.random.default_rng(1)
    true = rng.dirichlet([4, 4, 4], size=60)
    pred = np.clip(true + rng.normal(0, 0.05, true.shape), 0, 1)
    pred /= pred.sum(axis=1, keepdims=True)
    return true, pred


# --------------------------------------------------------------------------- #
# DiagonalDisplay
# --------------------------------------------------------------------------- #
def test_diagonal_from_predictions(protocol_predictions):
    true, pred = protocol_predictions
    disp = DiagonalDisplay.from_predictions(true, pred)
    assert isinstance(disp.ax_, plt.Axes)
    assert disp.figure_ is disp.ax_.figure
    assert disp.line_ is not None  # diagonal drawn


def test_diagonal_uses_given_ax_and_kwargs(protocol_predictions):
    true, pred = protocol_predictions
    _, ax = plt.subplots()
    disp = DiagonalDisplay.from_predictions(true, pred, ax=ax, color="red")
    assert disp.ax_ is ax
    # **kwargs forwarded to scatter
    assert np.allclose(disp.scatter_.get_facecolor()[0][:3], (1.0, 0.0, 0.0))


def test_diagonal_multiclass_draws_all_classes(protocol_predictions_3):
    true, pred = protocol_predictions_3
    disp = DiagonalDisplay.from_predictions(true, pred)
    assert isinstance(disp.scatter_, list) and len(disp.scatter_) == 3


def test_diagonal_from_protocol(binary_dataset):
    X, y = binary_dataset
    disp = DiagonalDisplay.from_protocol(
        CC(LogisticRegression()), X, y, n_prevalences=5, batch_size=50,
        random_state=0,
    )
    assert isinstance(disp.ax_, plt.Axes)


# --------------------------------------------------------------------------- #
# BiasDisplay
# --------------------------------------------------------------------------- #
def test_bias_global(protocol_predictions):
    true, pred = protocol_predictions
    disp = BiasDisplay.from_predictions(true, pred)
    assert "boxes" in disp.boxplot_
    assert len(disp.boxplot_["boxes"]) == 2  # one box per class


def test_bias_binned(protocol_predictions):
    true, pred = protocol_predictions
    disp = BiasDisplay.from_predictions(true, pred, bins=4)
    assert 0 < len(disp.boxplot_["boxes"]) <= 4


# --------------------------------------------------------------------------- #
# ErrorByShiftDisplay
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", ["ae", "se", "rae"])
def test_error_by_shift_metrics(protocol_predictions, metric):
    true, pred = protocol_predictions
    disp = ErrorByShiftDisplay.from_predictions(
        true, pred, error_metric=metric, n_bins=5, name="CC"
    )
    assert disp.line_ is not None
    assert disp.shift_.shape == (len(true),)
    assert disp.error_.shape == (len(true),)


def test_error_by_shift_bad_metric(protocol_predictions):
    true, pred = protocol_predictions
    with pytest.raises(ValueError):
        ErrorByShiftDisplay.from_predictions(true, pred, error_metric="nope")


# --------------------------------------------------------------------------- #
# PrevalenceDisplay
# --------------------------------------------------------------------------- #
def test_prevalence_from_predictions_with_true():
    disp = PrevalenceDisplay.from_predictions(
        [0.3, 0.7], true_prevalence=[0.4, 0.6], class_names=["neg", "pos"]
    )
    assert disp.true_bar_ is not None
    assert len(disp.bar_) == 2


def test_prevalence_accepts_dict():
    disp = PrevalenceDisplay.from_predictions({"a": 0.2, "b": 0.8})
    assert disp.true_bar_ is None
    np.testing.assert_allclose(disp.predicted_prevalence, [0.2, 0.8])


def test_prevalence_from_estimator(binary_dataset):
    X, y = binary_dataset
    q = CC(LogisticRegression()).fit(X, y)
    disp = PrevalenceDisplay.from_estimator(q, X)
    assert isinstance(disp.ax_, plt.Axes)


# --------------------------------------------------------------------------- #
# ConfidenceRegionDisplay
# --------------------------------------------------------------------------- #
def test_confidence_interval_layout():
    rng = np.random.default_rng(0)
    estims = rng.dirichlet([6, 6], size=200)
    disp = ConfidenceRegionDisplay.from_estimates(estims, class_names=["a", "b"])
    assert disp.errorbar_ is not None
    assert disp.ellipse_ is None


def test_confidence_ternary_ellipse():
    rng = np.random.default_rng(0)
    estims = rng.dirichlet([8, 6, 6], size=300)
    disp = ConfidenceRegionDisplay.from_estimates(
        estims, class_names=["a", "b", "c"], true_prevalence=[0.4, 0.3, 0.3]
    )
    assert disp.ellipse_ is not None
    assert disp.scatter_ is not None


def test_confidence_from_region():
    rng = np.random.default_rng(0)
    estims = rng.dirichlet([6, 6, 6], size=200)
    region = construct_confidence_region(estims, confidence_level=0.9, method="ellipse")
    disp = ConfidenceRegionDisplay.from_region(region)
    assert disp.confidence_level == 0.9
    assert disp.ellipse_ is not None


def test_confidence_ternary_forwards_style_kwargs():
    # Regression: passing alpha/s must not collide with the artist defaults.
    rng = np.random.default_rng(0)
    estims = rng.dirichlet([8, 6, 6], size=200)
    disp = ConfidenceRegionDisplay.from_estimates(
        estims, class_names=["a", "b", "c"], color="#1d3557", alpha=0.25, s=8
    )
    assert disp.scatter_ is not None


def test_confidence_interval_forwards_style_kwargs():
    rng = np.random.default_rng(0)
    estims = rng.dirichlet([6, 6], size=200)
    disp = ConfidenceRegionDisplay.from_estimates(estims, capsize=6, color="C2")
    assert disp.errorbar_ is not None


def test_prevalence_forwards_style_kwargs():
    disp = PrevalenceDisplay.from_predictions([0.3, 0.7], capsize=6, color="C1")
    assert len(disp.bar_) == 2


def test_ternary_requires_three_classes():
    rng = np.random.default_rng(0)
    estims = rng.dirichlet([6, 6], size=100)
    with pytest.raises(ValueError):
        ConfidenceRegionDisplay.from_estimates(estims, kind="ternary")
