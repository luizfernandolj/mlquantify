import numpy as np
import pytest

from mlquantify.representations import (
    DistanceRepresentation,
    HistogramRepresentation,
    KDERepresentation,
    KernelMeanRepresentation,
    PredictionRepresentation,
)


def test_histogram_representation_fits_classwise_histograms():
    X = np.array([[0.1, 0.2], [0.2, 0.4], [0.8, 0.6], [0.9, 0.7]])
    y = np.array([0, 0, 1, 1])

    representation = HistogramRepresentation(bins=[2, 4], mode="histogram")
    representation.fit(X, y)

    transformed = representation.transform(X)

    assert representation.classes_.tolist() == [0, 1]
    assert representation.class_representations_.shape == (2, transformed.shape[0])
    np.testing.assert_allclose(
        representation.class_representations_[0],
        representation.transform(X[y == 0]),
    )


def test_prediction_representation_supports_average_and_sample_modes():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2], [0.2, 0.8]])
    y = np.array([0, 0, 1, 1])

    average = PredictionRepresentation(method="soft", average=True)
    average.fit(X, y)

    samples = PredictionRepresentation(
        func=lambda X, representation: np.asarray(X, dtype=float)[:, -1],
        average=False,
    )
    samples.fit(X, y)

    np.testing.assert_allclose(average.transform(X), X.mean(axis=0))
    assert average.class_representations_.shape == (2, 2)
    assert samples.class_representations_.shape == (2,)
    np.testing.assert_allclose(samples.class_representations_[1], [0.2, 0.8])


def test_prediction_representation_accepts_custom_functions():
    X = np.array([[0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1])

    func = PredictionRepresentation(
        func=lambda X, representation: np.asarray(X, dtype=float)[:, -1],
        average=True,
    )

    func.fit(X, y)

    assert func.transform(X) == np.mean(X[:, -1])


def test_prediction_representation_accepts_only_soft_and_hard_methods():
    X = np.array([[0.6, 0.4], [0.3, 0.7]])
    y = np.array([0, 1])

    with pytest.raises(ValueError, match="Unknown prediction representation method"):
        PredictionRepresentation(method="custom").fit(X, y)


def test_prediction_representation_hard_mode():
    y = np.array([0, 0, 1, 1])
    labels = np.array([0, 1, 1, 0])

    hard = PredictionRepresentation(method="hard", average=True)
    hard.fit(labels, y)

    np.testing.assert_allclose(hard.transform(labels), [0.5, 0.5])


def test_kde_representation_exposes_class_likelihoods():
    X = np.array([[0.0], [0.1], [0.9], [1.0]])
    y = np.array([0, 0, 1, 1])

    representation = KDERepresentation(bandwidth=0.2)
    representation.fit(X, y)

    likelihoods = representation.class_likelihoods(X)

    assert representation.class_representations_.shape == (2,)
    assert likelihoods.shape == (2, 4)
    assert np.all(likelihoods > 0)


def test_distance_representation_fits_classwise_mean_distances():
    X = np.array([[0.0], [2.0], [10.0], [14.0]])
    y = np.array([0, 0, 1, 1])

    representation = DistanceRepresentation(metric="euclidean")
    representation.fit(X, y)

    np.testing.assert_allclose(representation.transform(X[y == 0]), [1.0, 11.0])
    np.testing.assert_allclose(representation.transform(X[y == 1]), [11.0, 2.0])
    assert representation.class_representations_.shape == (2, 2)
    np.testing.assert_allclose(
        representation.class_representations_[0],
        representation.transform(X[y == 0]),
    )


def test_kernel_mean_representation_and_pairwise_kernel():
    X = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]])
    y = np.array([0, 0, 1, 1])

    representation = KernelMeanRepresentation(kernel="linear")
    representation.fit(X, y)

    np.testing.assert_allclose(representation.transform(X), X.mean(axis=0))
    assert representation.class_representations_.shape == (2, 2)
    assert representation.pairwise(X[:2], X[2:]).shape == (2, 2)
