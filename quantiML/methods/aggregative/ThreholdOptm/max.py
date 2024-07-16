import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class MAX(ThresholdOptimization):
    """ Implementation of MAX
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner, threshold)
    
    
    def best_tprfpr(self, threshold:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        max_index = np.argmax(np.abs(tprs - fprs))
        tpr, fpr = tprs[max_index], fprs[max_index]
        return (tpr, fpr)