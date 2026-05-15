import numpy as np
from sklearn.metrics import pairwise_kernels

class ScoreRepresentation:
    def fit(self, X, y, classes=None):
        self.classes_ = np.asarray(classes) if classes is not None else np.unique(y)
        self.train_scores_ = self._positive_scores(X)
        self.train_labels_ = np.asarray(y)
        return self

    def transform(self, X):
        return self._positive_scores(X)

    def split_train(self):
        neg = self.train_scores_[self.train_labels_ == self.classes_[0]]
        pos = self.train_scores_[self.train_labels_ == self.classes_[1]]
        return pos, neg

    def _positive_scores(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 2:
            return X[:, 1]
        return X.ravel()


class HistogramRepresentation:
    def __init__(
        self,
        bins=(10, 20, 30),
        mode="histogram",
    ):
        self.bins = bins
        self.mode = mode
    
    @property
    def bins_(self):
        if np.isscalar(self.bins):
            return np.asarray([int(self.bins)])
        return np.asarray(self.bins, dtype=int)

    def fit(self, X, y):
        X = self._as_2d(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)

        self.class_representations_ = np.asarray([
            self.transform(X[y == cls])
            for cls in self.classes_
        ])

        return self

    def transform(self, X):
        X = self._as_2d(X)

        if self.mode == "histogram":
            return self._histogram_matrix(X)

        if self.mode == "onehot":
            return self._onehot_histogram_matrix(X)

        raise ValueError(f"Unknown mode {self.mode!r}")

    def _histogram_matrix(self, X):
        """
        Original HDy/HDx representation.

        Returns one normalized histogram per feature/bin-size.
        """
        histograms = []

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]

            for n_bins in self.bins_:
                hist, _ = np.histogram(
                    values,
                    bins=int(n_bins),
                    range=(0, 1),
                    density=False,
                )

                hist = hist.astype(float)
                hist /= max(hist.sum(), 1.0)

                histograms.append(hist)

        return np.concatenate(histograms)

    def _onehot_histogram_matrix(self, X):
        """
        Generalized histogram representation.

        Each instance is mapped to a one-hot bin encoding,
        then averaged across samples.

        Used for q = Mp formulations.
        """
        representations = []

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]

            for n_bins in self.bins_:
                edges = np.linspace(0.0, 1.0, int(n_bins) + 1)

                # bin index per sample
                bin_ids = np.digitize(values, edges[1:-1], right=False)

                # one-hot encoding
                onehot = np.zeros((len(values), int(n_bins)), dtype=float)
                onehot[np.arange(len(values)), bin_ids] = 1.0

                # average representation
                representations.append(onehot.mean(axis=0))

        return np.concatenate(representations)

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            return X.reshape(-1, 1)

        return X
    

class KDERepresentation:
    def __init__(self, bandwidth=0.1, kernel="gaussian"):
        self.bandwidth = bandwidth
        self.kernel = kernel

    def fit(self, X, y, classes=None, sample_weight=None):
        X = self._as_2d(X)
        y = np.asarray(y)
        self.classes_ = np.asarray(classes) if classes is not None else np.unique(y)

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

        return self

    def transform(self, X):
        return self._as_2d(X)

    def class_likelihoods(self, X):
        X = self._as_2d(X)
        return np.asarray([
            np.exp(kde.score_samples(X)) + 1e-12
            for kde in self.class_kdes_
        ])

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X
    


class KernelMeanRepresentation:
    def __init__(self, kernel="rbf", gamma=None, degree=3, coef0=0.0):
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y, classes=None, sample_weight=None):
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self.classes_ = np.asarray(classes) if classes is not None else np.unique(y)

        K = pairwise_kernels(
            self.X_train_,
            self.X_train_,
            metric=self.kernel,
            **self._kernel_kwargs(),
        )

        self.class_means_ = []
        for cls in self.classes_:
            mask = self.y_train_ == cls
            if sample_weight is None:
                self.class_means_.append(K[mask].mean(axis=0))
            else:
                w = np.asarray(sample_weight)[mask]
                self.class_means_.append(np.average(K[mask], axis=0, weights=w))

        self.class_means_ = np.vstack(self.class_means_)
        return self

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