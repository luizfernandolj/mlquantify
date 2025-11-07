import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.base_aggregative import (
    AggregationMixin,
    _get_learner_function
)
from mlquantify.adjust_counting import CC
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_predictions, validate_y, validate_data, validate_prevalences



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
        
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        return tags
    
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        validate_y(self, y)
        self.classes = np.unique(y)

        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes])
        self.priors = counts / len(y)
        self.y_train = y
                
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        estimator_function = _get_learner_function(self)
        predictions = getattr(self.learner, estimator_function)(X)
        prevalences = self.aggregate(predictions, y_train=self.y_train)
        return prevalences

    def aggregate(self, predictions, y_train=None):
        predictions = validate_predictions(self, predictions)
        if not hasattr(self, 'priors'):
            if y_train is None:
                raise ValueError("y_train must be provided if the quantifier is not fitted.")
            self.classes = np.unique(y_train)
            counts = np.array([np.count_nonzero(y_train == _class) for _class in self.classes])
            self.priors = counts / len(y_train)
        
        prevalences = self._iterate(predictions, self.priors)
        prevalences = validate_prevalences(self, prevalences, self.classes)
        return prevalences
    
    
    def _iterate(self, predictions, priors):
        ...
