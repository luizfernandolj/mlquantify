import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.base_aggregative import (
    AggregationMixin,
    SoftLearnerQMixin,
    _get_learner_function
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_y
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._constraints import (
    Options,
    Interval,
    CallableConstraint
)
from sklearn.neighbors import KernelDensity


class BaseMixture(BaseQuantifier):
    """Base class for mixture-based quantifiers."""
    
    def __init__(self, learner=None):
        super().__init__(learner)
        self._precomputed = False
        self.distances = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
        validate_y(self, y)
        self.classes = np.unique(y)
        
        self._precompute_training(X, y, *args, **kwargs)
        self._precomputed = True
        
        return self
    

    def predict(self, X):
        """Predict class prevalences for the given data."""
        predictions = getattr(self.learner, _get_learner_function(self))(X)

        prevalences = self.aggregate(predictions, self.train_predictions, self.train_y_values)
        return prevalences
    
    def aggregate(self, predictions, train_predictions, train_y_values):
        
        self.classes = self.classes if hasattr(self, 'classes') else np.unique(train_y_values)
        
        if not self._precomputed:
            self._precompute_training(train_predictions, train_y_values)
            self._precomputed = True
        
        prevalence, _ = self._best_mixture(predictions, train_predictions, train_y_values)

        return prevalence
    
    @abstractmethod
    def _mixture(self, predictions, models):
        ...
        
    @abstractmethod
    def _precompute_training(self, train_predictions, train_y_values):
        ...