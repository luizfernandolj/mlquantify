
import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class ACC(ThresholdOptimization):
    """ Implementation of Adjusted Classify and Count
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner, threshold)
    
    
    def best_tprfpr(self, threshold:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        tpr = tprs[threshold == self.threshold][0]
        fpr = fprs[threshold == self.threshold][0]
        return (tpr, fpr)