import numpy as np     
from abc import abstractmethod
from mlquantify.adjust_counting._counting import CC
from mlquantify.base import (
    BaseQuantifier, 
    BinaryQMixin,
    uses_soft_predictions,
    _get_learner_function,
)
from mlquantify.base_aggregative import (
    AggregationMixin,
    uses_soft_predictions,
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_y
from mlquantify.utils._get_scores import apply_cross_validation

class BaseAdjustCount(AggregationMixin, BaseQuantifier):
    """Base class for adjustment-based count quantifiers."""
    
    def __init__(self, learner=None):
        self.learner = learner

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
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

        predictions = _get_learner_function(self)(X)
        
        prevalences = self.aggregate(predictions, self.train_predictions, self.train_y_values)
        return prevalences

    def aggregate(self, predictions, *args):
        return self._adjust(predictions, *args)


class BaseThresholdAdjustment(BaseAdjustCount):
    
    def _adjust(self, predictions, train_y_scores, train_y_values):
        
        # get tpr and fpr values, along with threholds
        
        # get best threshold based on some criterion (method's speciftc)
        
        # get predictions for CC
        
        # Compute equation of threshold methods to compute prevalence
        
        # return prevalence
        pass

class BaseMatrixAdjustment(BaseAdjustCount): # FM, GAC, GPAC
    
    def _adjust(self, predictions, train_y_pred, train_y_values):
        
        # compute confusion matrix
        
        # applying the adjustment type (linear system, matrix inversion, etc)
        
        # return prevalence
        pass