import numpy as np
from scipy.spatial.distance import cdist

from mlquantify.representations._base import BaseRepresentation


class DistanceRepresentation(BaseRepresentation):
    def __init__(self, metric="euclidean"):
        self.metric = metric

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        values = np.zeros((X.shape[0], len(self.classes_)), dtype=float)

        for class_idx, cls in enumerate(self.classes_):
            X_cls = self.X_train_[self.y_train_ == cls]

            if len(X_cls) > 0:
                values[:, class_idx] = cdist(
                    X,
                    X_cls,
                    metric=self.metric,
                ).mean(axis=1)

        return values.mean(axis=0)

    def _fit(self, X, y, sample_weight=None):
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y)

        self.class_representations_ = np.asarray([
            self.transform(self.X_train_[self.y_train_ == cls])
            for cls in self.classes_
        ])
