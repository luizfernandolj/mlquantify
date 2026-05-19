import numpy as np
import pytest

from mlquantify.losses import (
    DistanceLoss,
    EnergyLoss,
    HellingerSurrogateLoss,
    LeastSquaresLoss,
    MixtureNegativeLogLikelihoodLoss,
    NegativeLogLikelihoodLoss,
    RegularizedMixtureNLLLoss,
    get_loss,
)
from mlquantify.metrics import hellinger, sqEuclidean


def test_distance_loss_matches_metric_and_normalizes():
    mixture = np.array([2.0, 2.0])
    target = np.array([1.0, 3.0])

    loss = DistanceLoss(distance="hellinger", normalize=True)

    assert loss(mixture, target) == pytest.approx(
        hellinger(np.array([0.5, 0.5]), np.array([0.25, 0.75]))
    )


def test_least_squares_and_factory_aliases():
    mixture = np.array([0.2, 0.8])
    target = np.array([0.5, 0.5])

    assert LeastSquaresLoss()(mixture, target) == pytest.approx(0.18)
    assert get_loss("sqEuclidean", normalize=False)(mixture, target) == pytest.approx(
        sqEuclidean(mixture, target)
    )
    assert isinstance(get_loss("ls"), LeastSquaresLoss)
    assert isinstance(get_loss("least_squares"), LeastSquaresLoss)
    assert isinstance(get_loss("least-squares"), LeastSquaresLoss)
    assert isinstance(get_loss("least squares"), LeastSquaresLoss)


def test_callable_loss_is_returned_unchanged():
    def custom_loss(a, b):
        return 7.0

    assert get_loss(custom_loss) is custom_loss
    assert get_loss(custom_loss)(None, None) == 7.0


def test_hellinger_surrogate_is_negative_overlap_objective():
    loss = HellingerSurrogateLoss(normalize=False)

    assert loss([0.25, 0.75], [0.25, 0.75]) == pytest.approx(-1.0)


def test_energy_loss_matches_quadratic_form():
    prevalence = np.array([0.7, 0.3])
    target_distances = np.array([1.0, 3.0])
    class_distances = np.array([[0.5, 2.0], [2.0, 1.0]])

    expected = prevalence @ (
        2.0 * target_distances - class_distances @ prevalence
    )

    assert EnergyLoss()(prevalence, target_distances, class_distances) == pytest.approx(
        expected
    )
    assert isinstance(get_loss("energy"), EnergyLoss)
    assert isinstance(get_loss("energy_distance"), EnergyLoss)


def test_negative_log_likelihood_reductions():
    likelihood = np.array([0.5, 0.25])

    assert NegativeLogLikelihoodLoss(reduction="mean")(likelihood) == pytest.approx(
        -np.log(likelihood).mean()
    )
    assert NegativeLogLikelihoodLoss(reduction="sum")(likelihood) == pytest.approx(
        -np.log(likelihood).sum()
    )


def test_mixture_nll_accepts_both_likelihood_orientations():
    prevalences = np.array([0.4, 0.6])
    class_likelihoods = np.array([[0.2, 0.8, 0.5], [0.7, 0.3, 0.4]])
    expected = -np.log(prevalences @ class_likelihoods).mean()

    loss = MixtureNegativeLogLikelihoodLoss()

    assert loss(prevalences, class_likelihoods) == pytest.approx(expected)
    assert loss(prevalences, class_likelihoods.T) == pytest.approx(expected)


def test_regularized_mixture_nll_adds_smoothness_penalties():
    prevalences = np.array([0.2, 0.3, 0.5])
    class_likelihoods = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.2],
            [0.1, 0.1, 0.7],
        ]
    )

    base = MixtureNegativeLogLikelihoodLoss()(prevalences, class_likelihoods)
    regularized = RegularizedMixtureNLLLoss(tau_0=0.5, tau_1=0.25)(
        prevalences,
        class_likelihoods,
    )

    assert regularized > base
