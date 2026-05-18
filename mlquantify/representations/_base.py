

from abc import ABC, abstractmethod
import numpy as np

class BaseRepresentation(ABC):
    """Base class for quantification representations."""

    def fit(self, X, y, classes=None, sample_weight=None):
        X = self._validate_X(X)
        y = np.asarray(y)

        self.classes_ = np.asarray(classes) if classes is not None else np.unique(y)
        self._fit(X, y, sample_weight=sample_weight)

        if not hasattr(self, "class_representations_"):
            raise AttributeError(
                f"{type(self).__name__} must define class_representations_ in _fit()."
            )

        return self

    @abstractmethod
    def transform(self, X):
        """Transform data into the representation space."""

    @abstractmethod
    def _fit(self, X, y, sample_weight=None):
        """Fit representation internals."""

    def _validate_X(self, X):
        return self._as_2d(X)

    @staticmethod
    def _as_2d(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            return X.reshape(-1, 1)
        return X