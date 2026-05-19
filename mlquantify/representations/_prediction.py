import numpy as np

from mlquantify.representations._base import BaseRepresentation


class PredictionRepresentation(BaseRepresentation):
    def __init__(self, method="soft", average=True, func=None, representation=None):
        self.method = method
        self.average = average
        self.func = func
        self.representation = representation

    def transform(self, X):
        Z = self._transformer()(X, self)

        if self.representation is not None:
            return self.representation.transform(Z)

        return self._aggregate(Z)

    def _aggregate(self, Z):
        if self.average:
            return np.mean(Z, axis=0)

        return Z

    def _fit(self, X, y, sample_weight=None):
        self.priors_ = np.asarray([
            np.mean(y == cls)
            for cls in self.classes_
        ])

        Z = self._transformer()(X, self)

        if self.representation is not None:
            self.representation.fit(
                Z,
                y,
                classes=self.classes_,
                sample_weight=sample_weight,
            )
            self.class_representations_ = self.representation.class_representations_
            return

        if self.average:
            self.class_representations_ = np.asarray([
                self._aggregate(Z[y == cls])
                for cls in self.classes_
            ], dtype=float)
        else:
            self.class_representations_ = np.empty(len(self.classes_), dtype=object)

            for class_idx, cls in enumerate(self.classes_):
                self.class_representations_[class_idx] = self._aggregate(Z[y == cls])

    def _transformer(self):
        if self.func is not None:
            return self.func

        transformers = {
            "soft": self._soft_predictions,
            "hard": self._hard_predictions,
        }

        try:
            return transformers[self.method]
        except KeyError as exc:
            raise ValueError(
                f"Unknown prediction representation method {self.method!r}. "
                f"Expected one of {sorted(transformers)} or pass func=..."
            ) from exc

    @staticmethod
    def _soft_predictions(X, representation):
        return np.asarray(X, dtype=float)

    @staticmethod
    def _hard_predictions(X, representation):
        X = np.asarray(X)

        if X.ndim == 2:
            label_indices = np.argmax(X, axis=1)
            labels = representation.classes_[label_indices]
        else:
            labels = X

        onehot = np.zeros((len(labels), len(representation.classes_)), dtype=float)

        for class_idx, cls in enumerate(representation.classes_):
            onehot[:, class_idx] = labels == cls

        return onehot

    def class_likelihoods(self, X):
        if self.representation is None or not hasattr(self.representation, "class_likelihoods"):
            raise AttributeError(
                f"{type(self).__name__} does not expose class likelihoods without "
                "a nested likelihood representation."
            )

        return self.representation.class_likelihoods(X)


class HardPredictionRepresentation(PredictionRepresentation):
    def __init__(self, average=True):
        super().__init__(method="hard", average=average)


class SoftPredictionRepresentation(PredictionRepresentation):
    def __init__(self, average=True):
        super().__init__(method="soft", average=average)
