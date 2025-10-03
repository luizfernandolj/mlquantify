"""
This module provides the classes of different types of aggregative quantifiers.
"""

import numpy as np

from mlquantify.aggregation._base import (
    AggregativeQuantifier,
    BinaryAggregativeQuantifier,
)

from mlquantify.base import (
    CrispLearnerQMixin,
    SoftLearnerQMixin
)

class CC(CrispLearnerQMixin, AggregativeQuantifier):

    def __init__(self, learner=None, threshold=0.5):
        super().__init__(learner=learner)
        self.threshold = threshold
    
    def fit(self, X, y, fitted_learner=False, *args, **kwargs):
        return self._fit(X, y, fitted_learner=fitted_learner, *args, **kwargs)
         
    def aggregate(self, predictions):

        predictions = self._get_valid_predictions(predictions, self.threshold)
        
        self.classes = np.unique(predictions)
        
        class_counts = np.array([np.count_nonzero(predictions == _class) for _class in self.classes])
      
        # Calculate the prevalence of each class
        prevalences = class_counts / len(predictions)
        
        return prevalences
        
    
    
    
    
    
    
