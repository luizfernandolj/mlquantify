    

from mlquantify.representations._base import BaseRepresentation
import numpy as np


class ScoreRepresentation(BaseRepresentation):
    def __init__(self, class_index=1):
        self.class_index = class_index

    def _fit(self, X, y, sample_weight=None):
        scores = self.transform(X)
        self.train_scores_ = scores
        self.train_labels_ = y

        self.class_representations_ = np.asarray(
            [scores[y == cls] for cls in self.classes_],
            dtype=object,
        )

    def transform(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim == 2:
            return X[:, self.class_index]

        return X.ravel()

    def split_train(self):
        neg = self.class_representations_[0]
        pos = self.class_representations_[1]
        return pos, neg