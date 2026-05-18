from mlquantify.representations._base import BaseRepresentation
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

class KernelMeanRepresentation(BaseRepresentation):
    def __init__(self, kernel="rbf", gamma=None, degree=3, coef0=0.0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _fit(self, X, y, sample_weight=None):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)

        K = pairwise_kernels(
            self.X_train_,
            self.X_train_,
            metric=self.kernel,
            **self._kernel_kwargs(),
        )

        class_means = []

        for cls in self.classes_:
            mask = self.y_train_ == cls

            if sample_weight is None:
                class_means.append(K[mask].mean(axis=0))
            else:
                weights = np.asarray(sample_weight)[mask]
                class_means.append(np.average(K[mask], axis=0, weights=weights))

        self.class_representations_ = np.vstack(class_means)

    def transform(self, X):
        K_test_train = pairwise_kernels(
            np.asarray(X),
            self.X_train_,
            metric=self.kernel,
            **self._kernel_kwargs(),
        )

        return K_test_train.mean(axis=0)

    def _kernel_kwargs(self):
        params = {}

        if self.kernel == "rbf" and self.gamma is not None:
            params["gamma"] = self.gamma

        if self.kernel == "poly":
            params["degree"] = self.degree
            params["coef0"] = self.coef0

        if self.kernel == "sigmoid":
            if self.gamma is not None:
                params["gamma"] = self.gamma
            params["coef0"] = self.coef0

        return params
