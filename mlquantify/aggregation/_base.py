"""
This module provides the base classes for implementing aggregative type quantifiers and strategies.
"""

import numpy as np
import warnings
from abc import abstractmethod, ABCMeta
from mlquantify.base import (
    BaseQuantifier,
    BinaryQMixin,
    RegressorQMixin,
    uses_crisp_predictions,
)

from mlquantify.utils._tags import (
    get_tags
)


class AggregativeQuantifier(BaseQuantifier, metaclass=ABCMeta):
    """Base class for all aggregative quantifiers.
    """
    
    @abstractmethod
    def __init__(self, learner=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if learner is None:
            warnings.warn("No learner provided. It will be assumed that direct predictions will be passed by calling the aggregate method.")
        self.learner = learner
        self.train_scores = None
        self.y_train = None
        self.classes = None


    def _fit(self, X, y, fitted_learner=False, *args, **kwargs):
        self.classes = np.unique(y)
        if not fitted_learner and self.learner:
            self.learner.fit(X, y, *args, **kwargs)
        
        return self
        

    def predict(self, X, *args, **kwargs):
        test_predictions = self._apply_valid_learner_function(X, *args, **kwargs)

        if uses_crisp_predictions(self):
            return self.aggregate(test_predictions)
        else:
            return self.aggregate(test_predictions, train_scores=self.train_scores, y_train=self.y_train)
    
    @abstractmethod
    def aggregate(self, test_predictions, train_scores=None, y_train=None):
        ...
    
    def _apply_valid_learner_function(self, X, *args, **kwargs):
        tags = get_tags(self)
        
        function = tags.estimator_function
        estim_type = tags.estimator_type

        if function is None or estim_type is None:
            raise ValueError("Quantifier does not have a valid learner function or estimator type defined in its tags.") 
        
        if hasattr(self, function):
            if function == "predict_proba" and estim_type == "crisp":
                raise ValueError(f"Inconsistent tags: '{function}' function cannot be associated with 'crisp' estimator type.")
            
            predictions = getattr(self.learner, function)(X, *args, **kwargs)

        else:
            raise ValueError(f"Quantifier does not have the specified learner function: {function}.")
        
            
        return predictions
        
        
         
    def _get_valid_predictions(self, predictions, threshold=0.5):
        predictions = np.asarray(predictions)

        dimensions = predictions.shape[1] if len(predictions.shape) > 1 else 1
        
        if dimensions > 2:
            predictions = np.argmax(predictions, axis=1)
        elif dimensions == 2:
            predictions = (predictions[:, 1] > threshold).astype(int)
        elif dimensions == 1:
            if np.issubdtype(predictions.dtype, np.floating):
                predictions = (predictions > threshold).astype(int)
        else:
            raise ValueError(f"Predictions array has an invalid number of dimensions. Expected 1 or 2 dimensions, got {predictions.ndim}.")

        return predictions

    def set_params(self, **params):
        
        # Model Params
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Learner Params
        if self.learner is not None:
            learner_params = {k.replace('learner__', ''): v for k, v in params.items() if 'learner__' in k}
            if learner_params:
                self.learner.set_params(**learner_params)
        
        return self


class BinaryAggregativeQuantifier(AggregativeQuantifier, BinaryQMixin, metaclass=ABCMeta):
    
    @abstractmethod
    def __init__(self, learner=None, strategy="ova", *args, **kwargs):
        super().__init__(learner=learner, strategy=strategy, *args, **kwargs)
        
