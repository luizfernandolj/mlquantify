import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from mlquantify._config import config_context
from mlquantify.matching import (
    DyS,
    EDx,
    EDy,
    GHDy,
    HDx,
    HDy,
    KDEyCS,
    KDEyHD,
    KDEyML,
    MMD_RKHS,
    SORD,
)
from mlquantify.matching._generalized import KDEyML as GeneralizedKDEyML
from mlquantify.representations import PredictionRepresentation


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
        estimator=LogisticRegression(max_iter=1000, random_state=42),
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


def test_hdy_matches_median_grid_recipe():
    train_scores = np.asarray(
        [
            [0.96, 0.04],
            [0.86, 0.14],
            [0.72, 0.28],
            [0.32, 0.68],
            [0.14, 0.86],
            [0.05, 0.95],
        ]
    )
    y_train = np.asarray([0, 0, 0, 1, 1, 1])
    test_scores = np.asarray(
        [
            [0.92, 0.08],
            [0.78, 0.22],
            [0.40, 0.60],
            [0.18, 0.82],
            [0.08, 0.92],
        ]
    )
    bins_size = [2, 4]

    q = HDy(bins_size=bins_size)

    with config_context(prevalence_return_type="array"):
        prevalence = q.aggregate(test_scores, train_scores, y_train)

    expected = _hdy_median_grid_reference(test_scores, train_scores, y_train, bins_size)

    assert prevalence == pytest.approx(expected)


def _hdy_median_grid_reference(test_scores, train_scores, y_train, bins_size):
    train_positive = train_scores[:, 1]
    test_positive = test_scores[:, 1]
    estimates = []

    for bins in bins_size:
        neg_density = _hdy_train_hist(train_positive[y_train == 0], bins)
        pos_density = _hdy_train_hist(train_positive[y_train == 1], bins)
        test_density = np.histogram(
            test_positive,
            bins=bins,
            range=(0, 1),
            density=True,
        )[0]

        best_prev = None
        best_distance = None
        for prev in np.linspace(0.0, 1.0, 101):
            mixture = prev * pos_density + (1.0 - prev) * neg_density
            distance = np.sqrt(
                np.sum((np.sqrt(mixture) - np.sqrt(test_density)) ** 2)
            )
            if best_distance is None or distance < best_distance:
                best_prev = prev
                best_distance = distance

        estimates.append(best_prev)

    positive_prevalence = np.median(estimates)
    return np.asarray([1.0 - positive_prevalence, positive_prevalence])


def _hdy_train_hist(values, bins):
    hist = np.histogram(values, bins=bins, range=(0, 1), density=True)[0]
    return hist / hist.sum()


def test_kernel_matching_multiclass(multiclass_dataset):
    X, y = multiclass_dataset
    q = MMD_RKHS(kernel="rbf")

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)
    assert q.best_distance_ is not None


def test_energy_distance_matching_binary(binary_dataset):
    X, y = binary_dataset
    q = EDx()

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)
    assert q.best_distance_ is not None


def test_energy_distance_matching_aggregative_binary(binary_dataset):
    X, y = binary_dataset
    q = EDy(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=3,
    )

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=2)
    assert q.best_distance_ is not None


@pytest.mark.parametrize(
    "quantifier",
    [
        KDEyML(estimator=LogisticRegression(max_iter=1000, random_state=42), bandwidth=0.2),
        KDEyHD(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            bandwidth=0.2,
            montecarlo_trials=90,
            random_state=42,
        ),
        KDEyCS(estimator=LogisticRegression(max_iter=1000, random_state=42), bandwidth=0.2),
    ],
)
def test_kde_matching_multiclass(quantifier, multiclass_dataset):
    X, y = multiclass_dataset

    quantifier.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = quantifier.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)
    assert quantifier.best_distance_ is not None


def test_aggregative_generalized_matching_uses_prediction_representation():
    g_hdy = GHDy(estimator=LogisticRegression(max_iter=1000, random_state=42))
    kde_ml = GeneralizedKDEyML(estimator=LogisticRegression(max_iter=1000, random_state=42))

    assert isinstance(g_hdy.representation, PredictionRepresentation)
    assert g_hdy.representation.representation is not None
    assert isinstance(kde_ml.representation, PredictionRepresentation)
    assert kde_ml.representation.representation is not None


def test_sord_score_matching_binary(binary_dataset):
    X, y = binary_dataset
    q = SORD(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
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
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            bins_size=[5],
        )
    elif quantifier_class is SORD:
        q = quantifier_class(
            estimator=LogisticRegression(max_iter=1000, random_state=42),
            n_grid=21,
        )
    else:
        q = quantifier_class(bins_size=[5])

    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)

    _assert_valid_prevalence(prevalence, n_classes=3)
