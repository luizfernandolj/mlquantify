
import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class ACC(ThresholdOptimization):
    """ Adjusted Classify and Count or Adjusted Count. Is a 
    base method for the threhold methods.
        As described on the Threshold base class, this method 
    estimate the true positive and false positive rates from
    the training data and utilize them to adjust the output 
    of the CC method.
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        self.threshold = threshold
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        # Get the tpr and fpr where the threshold is equal to the base threshold, default is 0.5
        
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)