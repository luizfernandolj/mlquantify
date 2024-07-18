import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class MS2(ThresholdOptimization):
    """ Implementation of MAX
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        indices = np.where(np.abs(tprs - fprs) > 0.25)[0]
    
        threshold = np.median(thresholds[indices])
        tpr = np.median(tprs[indices])
        fpr = np.median(fprs[indices])
        
        return (threshold, tpr, fpr)