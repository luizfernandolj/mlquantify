

from abc import ABC, abstractmethod
import numpy as np

class BaseRepresentation(ABC):
    """Base class for quantification representations."""

    def fit(self, X, y, classes=None, sample_weight=None):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 0:
            raise ValueError("X must be array-like.")

        if y.ndim != 1:
            y = y.ravel()

        if len(X) != len(y):
            raise ValueError(
                f"X and y have inconsistent lengths: {len(X)} and {len(y)}."
            )

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
