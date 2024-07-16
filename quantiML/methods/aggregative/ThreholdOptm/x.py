import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class X_method(ThresholdOptimization):
    """ Implementation of MAX
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner, threshold)
    
    
    def best_tprfpr(self, threshold:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        min_index = np.argmin(np.abs(1 - (tprs + fprs)))
        tpr, fpr = tprs[min_index], fprs[min_index]
        return (tpr, fpr)