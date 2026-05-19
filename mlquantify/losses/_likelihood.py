# losses/_likelihood.py

import numpy as np

from mlquantify.losses._base import BaseLoss


EPS = 1e-12


def _reduce_negative_log_likelihood(likelihood, reduction):
    likelihood = np.asarray(likelihood, dtype=float)
    likelihood = np.maximum(likelihood, EPS)
    values = -np.log(likelihood)

    if reduction == "mean":
        return float(values.mean())

    if reduction == "sum":
        return float(values.sum())

    raise ValueError("reduction must be 'mean' or 'sum'.")


def _mixture_likelihood(prevalences, class_likelihoods):
    prevalences = np.asarray(prevalences, dtype=float)
    class_likelihoods = np.asarray(class_likelihoods, dtype=float)

    if class_likelihoods.ndim != 2:
        raise ValueError("class_likelihoods must be a 2D array.")

    if class_likelihoods.shape[0] == prevalences.shape[0]:
        return prevalences @ class_likelihoods

    if class_likelihoods.shape[1] == prevalences.shape[0]:
        return class_likelihoods @ prevalences

    raise ValueError(
        "class_likelihoods must have one dimension matching the number "
        "of prevalences."
    )


class NegativeLogLikelihoodLoss(BaseLoss):
    """Negative log-likelihood loss for mixture likelihoods."""

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, likelihood):
        return _reduce_negative_log_likelihood(likelihood, self.reduction)


class MixtureNegativeLogLikelihoodLoss(BaseLoss):
    """Negative log-likelihood for class likelihood mixtures."""

    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, prevalences, class_likelihoods):
        mixture = _mixture_likelihood(prevalences, class_likelihoods)
        return _reduce_negative_log_likelihood(mixture, self.reduction)


class RegularizedMixtureNLLLoss(BaseLoss):
    """Mixture NLL with optional ordinal smoothness regularization."""

    def __init__(self, tau_0=0.0, tau_1=0.0, reduction="mean"):
        self.tau_0 = tau_0
        self.tau_1 = tau_1
        self.reduction = reduction

    def __call__(self, prevalences, class_likelihoods):
        prevalences = np.asarray(prevalences, dtype=float)

        mixture = _mixture_likelihood(prevalences, class_likelihoods)
        loss = _reduce_negative_log_likelihood(mixture, self.reduction)

        if self.tau_0 > 0:
            loss += self.tau_0 * self._first_order_penalty(prevalences)

        if self.tau_1 > 0:
            loss += self.tau_1 * self._second_order_penalty(prevalences)

        return float(loss)

    @staticmethod
    def _first_order_penalty(prevalences):
        return np.sum((prevalences[1:] - prevalences[:-1]) ** 2) / 2.0

    @staticmethod
    def _second_order_penalty(prevalences):
        if len(prevalences) < 3:
            return 0.0

        return np.sum(
            (-prevalences[:-2] + 2.0 * prevalences[1:-1] - prevalences[2:]) ** 2
        ) / 2.0
