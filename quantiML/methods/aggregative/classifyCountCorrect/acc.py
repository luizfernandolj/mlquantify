
import numpy as np
from sklearn.base import BaseEstimator

from ....base import ThresholdOptimization

class ACC(ThresholdOptimization):
    """ Implementation of Adjusted Classify and Count
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner, threshold)
        
    
    def best_tprfpr(self, threshold:np.ndarray, tpr: np.ndarray, fpr: np.ndarray) -> tuple:
        tpr = tpr[threshold == self.threshold]
        fpr = fpr[threshold == self.threshold]
        return (tpr, fpr)