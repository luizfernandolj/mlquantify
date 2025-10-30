import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.base_aggregative import (
    AggregationMixin,
    _get_learner_function
)
from mlquantify.adjust_counting import CC
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_y



class BaseIterativeLikelihood(AggregationMixin, BaseQuantifier):
    """Base class for likelihood-based quantifiers."""

    @abstractmethod
    def __init__(self, 
                 learner=None,
                 tol=1e-4,
                 max_iter=100):
        self.learner = learner
        self.tol = tol
        self.max_iter = max_iter
        super().__init__()
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
        validate_y(self, y)
        self.classes = np.unique(y)

        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes])
        self.priors = counts / len(y)
                
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        estimator_function = _get_learner_function(self)
        predictions = getattr(self.learner, estimator_function)(X)
        prevalences = self.aggregate(predictions)
        return prevalences

    def aggregate(self, predictions, y_train=None):
        if not hasattr(self, 'priors'):
            self.classes = np.unique(y_train)
            counts = np.array([np.count_nonzero(y_train == _class) for _class in self.classes])
            self.priors = counts / len(y_train)
        return self._iterate(predictions, self.priors)
    
    
    def _iterate(self, predictions, priors):
        ...
