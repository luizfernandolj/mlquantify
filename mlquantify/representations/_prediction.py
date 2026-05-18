import numpy as np

from mlquantify.representations._base import BaseRepresentation


class HardPredictionRepresentation(BaseRepresentation):
    """Representation for hard classifier predictions.

    Used by GAC.
    """

    def transform(self, X):
        y_pred = np.asarray(X)

        if y_pred.ndim == 2:
            y_pred = self.classes_[np.argmax(y_pred, axis=1)]

        onehot = np.zeros((len(y_pred), len(self.classes_)), dtype=float)

        for idx, cls in enumerate(self.classes_):
            onehot[:, idx] = y_pred == cls

        return onehot.mean(axis=0)

    def _fit(self, X, y, sample_weight=None):
        X = np.asarray(X)

        self.class_representations_ = np.asarray([
            self.transform(X[y == cls])
            for cls in self.classes_
        ])


class SoftPredictionRepresentation(BaseRepresentation):
    def __init__(self, average=True):
        self.average = average

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "SoftPredictionRepresentation expects a 2D probability matrix."
            )

        if self.average:
            return X.mean(axis=0)

        return X

    def _fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)

        if self.average:
            self.class_representations_ = np.asarray([
                self.transform(X[y == cls])
                for cls in self.classes_
            ])
        else:
            self.train_predictions_ = X
            self.train_labels_ = np.asarray(y)
            self.class_representations_ = np.asarray([
                X[y == cls]
                for cls in self.classes_
            ], dtype=object)
