import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.mixture._utils import sqEuclidean
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_y, validate_data

from mlquantify.mixture._utils import (
    hellinger,
    topsoe,
    probsymm,
    sqEuclidean
)


class BaseMixture(BaseQuantifier):
    """Base class for mixture-based quantifiers."""
    
    def __init__(self):
        self._precomputed = False
        self.distances = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, 
                             X, y)
        validate_y(self, y)
        self.classes = np.unique(y)
        
        self._fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X, *args, **kwargs):
        """Predict class prevalences for the given data."""
        return self._predict(X, *args, **kwargs)
    
    def get_best_distance(self, *args, **kwargs):
        _, best_distance = self.best_mixture(*args, **kwargs)
        return best_distance

    @abstractmethod
    def best_mixture(self, X):
        """Determine the best mixture parameters for the given data."""
        pass
    
    @classmethod
    def get_distance(cls, dist_train, dist_test, measure="hellinger"):
        """
        Compute distance between two distributions.
        """
        
        if np.sum(dist_train) < 1e-20 or np.sum(dist_test) < 1e-20:
            raise ValueError("One or both vectors are zero (empty)...")
        if len(dist_train) != len(dist_test):
            raise ValueError("Arrays must have the same length.")

        dist_train = np.maximum(dist_train, 1e-20)
        dist_test = np.maximum(dist_test, 1e-20)

        if measure == "topsoe":
            return topsoe(dist_train, dist_test)
        elif measure == "probsymm":
            return probsymm(dist_train, dist_test)
        elif measure == "hellinger":
            return hellinger(dist_train, dist_test)
        elif measure == "euclidean":
            return sqEuclidean(dist_train, dist_test)
        else:
            raise ValueError(f"Invalid measure: {measure}")
    