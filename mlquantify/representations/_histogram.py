
from mlquantify.representations._base import BaseRepresentation
import numpy as np

class HistogramRepresentation(BaseRepresentation):
    def __init__(self, bins=(10, 20, 30), mode="histogram"):
        self.bins = bins
        self.mode = mode

    @property
    def bins_(self):
        if np.isscalar(self.bins):
            return np.asarray([int(self.bins)])
        return np.asarray(self.bins, dtype=int)

    def _fit(self, X, y, sample_weight=None):
        self.class_representations_ = np.asarray([
            self.transform(X[y == cls])
            for cls in self.classes_
        ])

    def transform(self, X):
        X = self._as_2d(X)

        if self.mode == "histogram":
            return self._histogram_matrix(X)

        if self.mode == "onehot":
            return self._onehot_histogram_matrix(X)

        raise ValueError(f"Unknown mode {self.mode!r}.")

    def _histogram_matrix(self, X):
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
        representations = []

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]

            for n_bins in self.bins_:
                edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
                bin_ids = np.digitize(values, edges[1:-1], right=False)

                onehot = np.zeros((len(values), int(n_bins)), dtype=float)
                onehot[np.arange(len(values)), bin_ids] = 1.0

                representations.append(onehot.mean(axis=0))

        return np.concatenate(representations)
