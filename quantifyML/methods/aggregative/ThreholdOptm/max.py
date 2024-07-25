import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class MAX(ThresholdOptimization):
    """ Implementation of MAX
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        max_index = np.argmax(np.abs(tprs - fprs))
        
        threshold = thresholds[max_index]
        tpr= tprs[max_index]
        fpr = fprs[max_index]
        return (threshold, tpr, fpr)