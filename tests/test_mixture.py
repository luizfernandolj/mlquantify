import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.matching import (
    DyS,
    EDx,
    EDy,
    HDx,
    HDy,
    MatchingHistogramQuantifier,
    MMD_RKHS,
    SMM,
    SORD,
)
from mlquantify.losses import DistanceLoss, LeastSquaresLoss


MATCHING_QUANTIFIERS = [DyS, HDy, SMM, SORD, HDx, MMD_RKHS, EDx, EDy]


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("quantifier_class", MATCHING_QUANTIFIERS)
def test_matching_methods_fit_predict_binary(quantifier_class, binary_dataset):
    X, y = binary_dataset

    if quantifier_class in (HDx, MMD_RKHS, EDx):
        q = quantifier_class()
    else:
        q = quantifier_class(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
        )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)


@pytest.mark.parametrize("quantifier_class", [DyS, HDy, HDx, SORD])
def test_binary_matching_methods_support_multiclass_ovr(
    quantifier_class,
    multiclass_dataset,
):
    X, y = multiclass_dataset

    if quantifier_class is HDx:
        q = quantifier_class(bins_size=[5])
    elif quantifier_class is SORD:
        q = quantifier_class(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_grid=21,
        )
    else:
        q = quantifier_class(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            bins_size=[5],
        )

    q.fit(X, y)

    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)


def test_histogram_matching_uses_distance_loss():
    q = HDx(bins_size=[5])

    assert isinstance(q.loss_function, DistanceLoss)
    assert q.loss_function.distance == "hellinger"
    assert q.loss_function.normalize is True


def test_histogram_matching_accepts_least_squares_alias():
    q = MatchingHistogramQuantifier(bins_size=[5], distance="ls", solver="grid")

    q._validate_params()

    assert isinstance(q.loss_function, LeastSquaresLoss)
