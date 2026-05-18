
from mlquantify.representations._base import BaseRepresentation
from sklearn.neighbors import KernelDensity
import numpy as np

EPS = 1e-10

class KDERepresentation(BaseRepresentation):
    def __init__(self, bandwidth=0.1, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def _fit(self, X, y, sample_weight=None):
        self.class_kdes_ = []

        for cls in self.classes_:
            mask = y == cls
            X_cls = X[mask]
            weights = None if sample_weight is None else np.asarray(sample_weight)[mask]

            if X_cls.shape[0] == 0:
                X_cls = np.ones((1, X.shape[1])) / X.shape[1]
                weights = None

            kde = KernelDensity(
                bandwidth=self.bandwidth,
                kernel=self.kernel,
            )
            kde.fit(X_cls, sample_weight=weights)

            self.class_kdes_.append(kde)

        self.class_representations_ = np.asarray(self.class_kdes_, dtype=object)

    def transform(self, X):
        return self._as_2d(X)

    def class_likelihoods(self, X):
        X = self.transform(X)

        return np.asarray([
            np.exp(kde.score_samples(X)) + EPS
            for kde in self.class_kdes_
        ])