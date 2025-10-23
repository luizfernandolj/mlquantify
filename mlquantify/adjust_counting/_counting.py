import numpy as np
from sklearn.naive_bayes import abstractmethod

from mlquantify.base_aggregative import (
    SoftLearnerQMixin,
    CrispLearnerQMixin
)

from mlquantify.adjust_counting._base import BaseCount
from mlquantify.utils._validation import validate_y, validate_predictions
from mlquantify.utils._constraints import Interval
        


class CC(CrispLearnerQMixin, BaseCount):
    
    
    _parameters_constraints = {
        "threshold": [
            Interval(0.0, 1.0),
            Interval(0, 1, discrete=True),
        ],
    }

    def __init__(self, learner=None, threshold=0.5):
        super().__init__(learner=learner)
        self.threshold = threshold

    def aggregate(self, predictions):
        predictions = self._get_valid_crisp_predictions(predictions, threshold=self.threshold)
        
        self.classes = self.classes if hasattr(self, 'classes') else np.unique(predictions)
        
        class_counts = np.array([np.count_nonzero(predictions == _class) for _class in self.classes])
        prevalences = class_counts / len(predictions)
        return prevalences
    
    
    def _get_valid_crisp_predictions(self, predictions, threshold=0.5):
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
            raise ValueError(f"Predictions array has an invalid number of dimensions. Expected 1 or more dimensions, got {predictions.ndim}.")

        return predictions


class PCC(SoftLearnerQMixin, BaseCount):

    def __init__(self, learner=None):
        super().__init__(learner=learner)

    def aggregate(self, predictions):
        validate_predictions(self, predictions)
        class_sums = np.sum(predictions, axis=0)
        prevalences = class_sums / len(predictions)
        if predictions.ndim == 1:
            prevalences = np.array([1-prevalences, prevalences])
        return prevalences