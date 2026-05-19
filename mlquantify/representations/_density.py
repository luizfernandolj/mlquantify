# representations/_density.py

import numpy as np

from sklearn.neighbors import KernelDensity

from ._base import BaseRepresentation


class KDERepresentation(BaseRepresentation):
    r"""Kernel density estimation representation."""

    def __init__(
        self,
        bandwidth=0.1,
        kernel="gaussian",
    ):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def transform(self, X):
        return np.asarray(X, dtype=float)

    def _fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)

        self.class_kdes_ = [
            KernelDensity(
                bandwidth=self.bandwidth,
                kernel=self.kernel,
            ).fit(X[y == cls])
            for cls in self.classes_
        ]

        self.class_representations_ = np.asarray(
            self.class_kdes_,
            dtype=object,
        )

    def class_likelihoods(self, X):
        X = np.asarray(X, dtype=float)

        return np.asarray([
            np.exp(kde.score_samples(X))
            for kde in self.class_kdes_
        ])