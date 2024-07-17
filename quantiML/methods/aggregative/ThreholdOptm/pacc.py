
import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization
from ....utils import adjust_threshold

class PACC(ThresholdOptimization):
    """ Implementation of Adjusted Classify and Count
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner, threshold)
    
    
    def _predict_method(self, X):
        prevalences = {}
        
        probabilities = self.learner.predict_proba(X)[:, 1]
        
        mean_scores = np.mean(probabilities)
        
        if self.tpr - self.fpr == 0:
            prevalence = mean_scores
        else:
            prevalence = (mean_scores - self.fpr) / (self.tpr - self.fpr)
        
        prevalences[self.classes[1]] = prevalence
        prevalences[self.classes[0]] = 1 - prevalence

        return prevalences
    
    
    def best_tprfpr(self, threshold:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        tpr = tprs[threshold == self.threshold][0]
        fpr = fprs[threshold == self.threshold][0]
        return (tpr, fpr)