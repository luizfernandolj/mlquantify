# representations/_kernel.py

import numpy as np

from sklearn.metrics.pairwise import pairwise_kernels

from ._base import BaseRepresentation


class KernelMeanRepresentation(BaseRepresentation):
    r"""Kernel mean embedding representation."""

    def __init__(
        self,
        kernel="rbf",
        gamma=None,
        degree=3,
        coef0=0.0,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        return X.mean(axis=0)

    def _fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)

        self.class_representations_ = np.asarray([
            self.transform(X[y == cls])
            for cls in self.classes_
        ])

    def pairwise(self, X, Y):
        params = {}

        if self.kernel in {"rbf", "poly", "sigmoid"} and self.gamma is not None:
            params["gamma"] = self.gamma

        if self.kernel == "poly":
            params["degree"] = self.degree
            params["coef0"] = self.coef0

        if self.kernel == "sigmoid":
            params["coef0"] = self.coef0

        return pairwise_kernels(
            X,
            Y,
            metric=self.kernel,
            **params,
        )
