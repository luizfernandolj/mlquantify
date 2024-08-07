import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class MS2(ThresholdOptimization):
    """ Median Sweep 2. It relies on the same
    strategy of the Median Sweep, but compute 
    the median only for cases in which 
    tpr -fpr > 0.25
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