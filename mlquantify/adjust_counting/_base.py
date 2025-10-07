import numpy as np

from mlquantify.base import (
    BaseQuantifier,
    AggregativeQuantifierMixin,
    _get_learner_function
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_y




class BaseCount(AggregativeQuantifierMixin, BaseQuantifier):
    """Base class for count-based quantifiers."""

    def __init__(self, learner=None):
        self.learner = learner

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, *args, **kwargs):
        """Fit the quantifier using the provided data and learner."""
        validate_y(self, y)
        self.classes = np.unique(y)
        if not learner_fitted:
            self.learner.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X):
        """Predict class prevalences for the given data."""
        predictions = _get_learner_function(self)(X)
        prevalences = self.aggregate(predictions)
        return prevalences
    
    @abstractmethod
    def aggregate(self, predictions):
        ...
    
