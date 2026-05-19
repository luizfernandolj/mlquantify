import numpy as np

from ._base import BaseRepresentation


class HistogramRepresentation(BaseRepresentation):
    r"""Histogram-based representation."""

    def __init__(
        self,
        bins=(10,),
        range=(0.0, 1.0),
        mode="histogram",
    ):
        self.bins = np.atleast_1d(bins)
        self.range = range
        self.mode = mode

    def transform(self, X):
        X = self._as_2d(X)

        histograms = []

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]

            for n_bins in self.bins:
                hist = self._compute_histogram(values, int(n_bins))
                histograms.append(hist)

        return np.concatenate(histograms)

    def _fit(self, X, y, sample_weight=None):
        X = self._as_2d(X)

        self.class_representations_ = np.asarray([
            self.transform(X[y == cls])
            for cls in self.classes_
        ])

    def _compute_histogram(self, values, bins):
        if self.mode == "histogram":
            hist, _ = np.histogram(
                values,
                bins=bins,
                range=self.range,
                density=False,
            )

            hist = hist.astype(float)
            hist /= max(hist.sum(), 1.0)

            return hist

        if self.mode == "onehot":
            edges = np.linspace(
                self.range[0],
                self.range[1],
                bins + 1,
            )

            indices = np.digitize(values, edges[1:-1], right=False)

            onehot = np.eye(bins)[indices]

            return onehot.mean(axis=0)

        raise ValueError(f"Unknown mode: {self.mode!r}")

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            return X.reshape(-1, 1)

        return X