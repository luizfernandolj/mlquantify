# losses/_distances.py

import numpy as np

from mlquantify.losses._base import BaseLoss
from mlquantify.metrics import (
    hellinger,
    topsoe,
    probsymm,
    sqEuclidean,
)


EPS = 1e-12


def normalize_distribution(x):
    x = np.asarray(x, dtype=float)
    x = np.maximum(x, EPS)

    total = x.sum()

    if total <= EPS:
        return np.ones_like(x) / len(x)

    return x / total


class DistanceLoss(BaseLoss):
    """Generic distance-based loss."""

    def __init__(self, distance="hellinger", normalize=True):
        self.distance = distance
        self.normalize = normalize

    def __call__(self, mixture, target):
        mixture = np.asarray(mixture, dtype=float)
        target = np.asarray(target, dtype=float)

        if self.normalize:
            mixture = normalize_distribution(mixture)
            target = normalize_distribution(target)

        if mixture.shape != target.shape:
            raise ValueError(
                f"mixture and target must have the same shape. "
                f"Got {mixture.shape} and {target.shape}."
            )

        if self.distance == "hellinger":
            return float(hellinger(mixture, target))

        if self.distance == "topsoe":
            return float(topsoe(mixture, target))

        if self.distance == "probsymm":
            return float(probsymm(mixture, target))

        if self.distance == "sqEuclidean":
            return float(sqEuclidean(mixture, target))

        if self.distance == "euclidean":
            return float(np.sqrt(sqEuclidean(mixture, target)))

        raise ValueError(f"Unknown distance: {self.distance!r}")


class LeastSquaresLoss(BaseLoss):
    """Squared Euclidean loss."""

    def __call__(self, mixture, target, M=None):
        mixture = np.asarray(mixture, dtype=float)
        target = np.asarray(target, dtype=float)

        if M is not None:
            mixture = np.asarray(M, dtype=float) @ mixture

        diff = target - mixture
        return float(diff @ diff)


class HellingerSurrogateLoss(BaseLoss):
    """Optimization surrogate for squared Hellinger distance."""

    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, mixture, target, M=None):
        mixture = np.asarray(mixture, dtype=float)
        target = np.asarray(target, dtype=float)

        if M is not None:
            mixture = np.asarray(M, dtype=float) @ mixture

        mixture = np.maximum(mixture, EPS)
        target = np.maximum(target, 0.0)

        if self.normalize:
            mixture = normalize_distribution(mixture)
            target = normalize_distribution(target)

        mask = target > 0

        return float(-np.sqrt(target[mask] * mixture[mask]).sum())


class EnergyLoss(BaseLoss):
    def __call__(self, prevalence, q, M):
        prevalence = np.asarray(prevalence, dtype=float)
        q = np.asarray(q, dtype=float)
        M = np.asarray(M, dtype=float)

        return float(prevalence @ (2.0 * q - M @ prevalence))
