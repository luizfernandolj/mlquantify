import numpy as np
from abc import abstractmethod

from mlquantify.base import BaseQuantifier

from mlquantify.base_aggregative import (
    AggregationMixin,
    _get_learner_function
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_predictions, validate_y, validate_data, validate_prevalences
from mlquantify.utils._get_scores import apply_cross_validation




class BaseCount(AggregationMixin, BaseQuantifier):
    """Base class for count-based quantifiers."""

    @abstractmethod
    def __init__(self, learner=None):
        self.learner = learner
        
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        tags.prediction_requirements.requires_train_labels = False
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        
        validate_y(self, y)
        
        self.classes = np.unique(y)
        if not learner_fitted:
            self.learner.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        estimator_function = _get_learner_function(self)
        predictions = getattr(self.learner, estimator_function)(X)
        prevalences = self.aggregate(predictions)
        return prevalences
    
    @abstractmethod
    def aggregate(self, predictions):
        ...
    


class BaseAdjustCount(AggregationMixin, BaseQuantifier):
    """Base class for adjustment-based count quantifiers."""
    
    @abstractmethod
    def __init__(self, learner=None):
        self.learner = learner

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False):
        """Fit the quantifier using the provided data and learner."""
        X, y = validate_data(self, X, y)
        
        validate_y(self, y)
        self.classes = np.unique(y)
        
        learner_function = _get_learner_function(self)
        
        if learner_fitted:
            train_predictions = getattr(self.learner, learner_function)(X)
            y_train_labels = y
        else:
            train_predictions, y_train_labels = apply_cross_validation(
                self.learner,
                X,
                y,
                function= learner_function,
                cv= 5,
                stratified= True,
                random_state= None,
                shuffle= True
            )
        
        self.train_predictions = train_predictions
        self.train_y_values = y_train_labels
                
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        predictions = getattr(self.learner, _get_learner_function(self))(X)
        
        prevalences = self.aggregate(predictions, self.train_predictions, self.train_y_values)
        return prevalences

    def aggregate(self, predictions, train_predictions, y_train_values):
        predictions = validate_predictions(self, train_predictions)
        prevalences = self._adjust(predictions, train_predictions, y_train_values)
        prevalences = validate_prevalences(self, prevalences, self.classes)
        return prevalences