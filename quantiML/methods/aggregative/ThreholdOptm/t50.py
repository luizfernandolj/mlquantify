import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class T50(ThresholdOptimization):
    """ Implementation of MAX
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner, threshold)
    
    
    def best_tprfpr(self, threshold:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        min_index = np.argmin(np.abs(tprs - 0.5))
        tpr = tprs[min_index]
        fpr = fprs[min_index]
        return (tpr, fpr)