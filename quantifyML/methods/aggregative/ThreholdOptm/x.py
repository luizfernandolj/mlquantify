import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class X_method(ThresholdOptimization):
    """ Threshold X. This method tries to
    use the threshold where fpr = 1 - tpr
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        min_index = np.argmin(abs(1 - (tprs + fprs)))
        
        threshold = thresholds[min_index]
        tpr = tprs[min_index]
        fpr = fprs[min_index]
        
        return (threshold, tpr, fpr)