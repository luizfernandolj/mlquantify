        
from abc import abstractmethod
from mlquantify.base import (
    AggregativeQuantifierMixin, 
    BaseQuantifier, 
    BinaryQMixin,
    uses_soft_predictions,
    _get_learner_function,
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_y
from mlquantify.utils._get_scores import apply_cross_validation

class BaseAdjustCount(AggregativeQuantifierMixin, BaseQuantifier):
    """Base class for adjustment-based count quantifiers."""
    
    def __init__(self, learner=None):
        self.learner = learner

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
        validate_y(self, y)
        self.classes = np.unique(y)
        
        if learner_fitted:
            train_predictions = _get_learner_function(self)(X)
            y_train_labels = y
        else:
            if uses_soft_predictions(self):
                train_predictions, y_train_labels = apply_cross_validation(
                    self.learner,
                    X,
                    y,
                    function= 'predict_proba',
                    cv= 5,
                    stratified= True,
                    random_state= None,
                    shuffle= True
                )
            else:
                train_predictions, y_train_labels = apply_cross_validation(
                    self.learner,
                    X,
                    y,
                    function= 'predict',
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
        ...

class BaseMatrixAdjustment(BaseAdjustCount):
    
    def _adjust(self, predictions, train_y_pred, train_y_values):
        ...
        
        