import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class MS(ThresholdOptimization):
    """ Implementation of MAX
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        self.threshold = threshold
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        tpr = np.median(tprs)
        fpr = np.median(fprs)
        return (self.threshold, tpr, fpr)