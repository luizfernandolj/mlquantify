import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.matching import DyS, HDx, HDy, KDEyCS, KDEyHD, KDEyML, MMD_RKHS, SORD


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("quantifier_class", [DyS, HDy])
def test_histogram_score_matching_binary(quantifier_class, binary_dataset):
    X, y = binary_dataset
    q = quantifier_class(
        learner=LogisticRegression(max_iter=1000, random_state=42),
        bins_size=[5, 10],
    )

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)
    assert q.best_distance_ is not None


def test_histogram_feature_matching_binary(binary_dataset):
    X, y = binary_dataset
    q = HDx(bins_size=[5, 10])

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)
    assert q.best_distance_ is not None


def test_kernel_matching_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = MMD_RKHS(kernel="rbf")

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)
    assert q.best_distance_ is not None


@pytest.mark.parametrize(
    "quantifier",
    [
        KDEyML(learner=LogisticRegression(max_iter=1000, random_state=42), bandwidth=0.2),
        KDEyHD(
            learner=LogisticRegression(max_iter=1000, random_state=42),
            bandwidth=0.2,
            montecarlo_trials=90,
            random_state=42,
        ),
        KDEyCS(learner=LogisticRegression(max_iter=1000, random_state=42), bandwidth=0.2),
    ],
)
def test_kde_matching_multiclass(quantifier, multiclass_dataset):
    X, y = multiclass_dataset

    quantifier.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = quantifier.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)
    assert quantifier.best_distance_ is not None


def test_sord_score_matching_binary(binary_dataset):
    X, y = binary_dataset
    q = SORD(
        learner=LogisticRegression(max_iter=1000, random_state=42),
        n_grid=21,
    )

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)
    assert q.best_distance_ is not None


@pytest.mark.parametrize("quantifier_class", [DyS, HDy, HDx, SORD])
def test_histogram_matching_multiclass_ovr(quantifier_class, multiclass_dataset):
    X, y = multiclass_dataset
    if quantifier_class in (DyS, HDy):
        q = quantifier_class(
            learner=LogisticRegression(max_iter=1000, random_state=42),
            bins_size=[5],
        )
    elif quantifier_class is SORD:
        q = quantifier_class(
            learner=LogisticRegression(max_iter=1000, random_state=42),
            n_grid=21,
        )
    else:
        q = quantifier_class(bins_size=[5])

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)
