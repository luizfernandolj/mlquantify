import numpy as np

from ._base import BaseRepresentation


class HistogramRepresentation(BaseRepresentation):
    r"""Histogram-based representation."""

    def __init__(
        self,
        bins=(10,),
        range=(0.0, 1.0),
        mode="histogram",
        features=None,
        partition_blocks=False,
        bin_edges="fixed",
    ):
        self.bins = np.atleast_1d(bins)
        self.range = range
        self.mode = mode
        self.features = features
        self.partition_blocks = partition_blocks
        self.bin_edges = bin_edges

    def transform(self, X):
        X = self._as_2d(X)
        X = self._select_features(X)

        histograms = []
        block_slices = []
        start = 0

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]

            for bin_idx, n_bins in enumerate(self.bins):
                hist = self._compute_histogram(
                    values,
                    int(n_bins),
                    edges=self._get_edges(feature_idx, bin_idx),
                )
                histograms.append(hist)

                stop = start + hist.shape[0]
                block_slices.append(slice(start, stop))
                start = stop

        representation = np.concatenate(histograms)

        if self.partition_blocks:
            self.block_slices_ = tuple(block_slices)

        return representation

    def _fit(self, X, y, sample_weight=None):
        X = self._as_2d(X)
        X_selected = self._select_features(X)

        self._fit_edges(X_selected)

        self.class_representations_ = np.asarray([
            self.transform(X[y == cls])
            for cls in self.classes_
        ])

    def _compute_histogram(self, values, bins, edges=None):
        if self.mode not in {"histogram", "onehot"}:
            raise ValueError(f"Unknown mode: {self.mode!r}")

        if edges is not None:
            hist_edges = np.asarray(edges, dtype=float).copy()
            hist_edges[0] = -np.inf
            hist_edges[-1] = np.inf
            hist, _ = np.histogram(
                values,
                bins=hist_edges,
                density=False,
            )

            hist = hist.astype(float)
            hist /= max(hist.sum(), 1.0)

            return hist

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

    def _fit_edges(self, X):
        if self.bin_edges == "fixed":
            return

        if self.bin_edges != "auto":
            raise ValueError("bin_edges must be either 'fixed' or 'auto'.")

        self.edges_ = []

        for feature_idx in range(X.shape[1]):
            values = X[:, feature_idx]
            feature_edges = []

            for n_bins in self.bins:
                feature_edges.append(
                    np.histogram_bin_edges(values, bins=int(n_bins))
                )

            self.edges_.append(feature_edges)

    def _get_edges(self, feature_idx, bin_idx):
        if self.bin_edges == "fixed":
            return None

        if self.bin_edges != "auto":
            raise ValueError("bin_edges must be either 'fixed' or 'auto'.")

        if not hasattr(self, "edges_"):
            raise AttributeError(
                "HistogramRepresentation must be fitted before transform() "
                "when bin_edges='auto'."
            )

        return self.edges_[feature_idx][bin_idx]

    def _select_features(self, X):
        if self.features is None:
            return X

        if isinstance(self.features, (int, np.integer)):
            feature = int(self.features)
            if X.shape[1] == 1 and feature == 1:
                return X
            return X[:, [feature]]

        return X[:, self.features]

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            return X.reshape(-1, 1)

        return X
