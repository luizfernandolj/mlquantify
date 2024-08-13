
import numpy as np
from sklearn.base import BaseEstimator

from ._ThreholdOptimization import ThresholdOptimization

class PACC(ThresholdOptimization):
    """ Probabilistic Adjusted Classify and Count. 
    This method adapts the AC approach by using average
    classconditional confidences from a probabilistic 
    classifier instead of true positive and false positive rates.
    """
    
    def __init__(self, learner:BaseEstimator, threshold:float=0.5):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        self.threshold = threshold
    
    def _predict_method(self, X):
        prevalences = {}
        
        probabilities = self.learner.predict_proba(X)[:, 1]
        
        mean_scores = np.mean(probabilities)
        
        if self.tpr - self.fpr == 0:
            prevalence = mean_scores
        else:
            prevalence = np.clip(abs(mean_scores - self.fpr) / (self.tpr - self.fpr), 0, 1)
        
        prevalences[self.classes[1]] = prevalence
        prevalences[self.classes[0]] = 1 - prevalence

        return prevalences
    
    
    
    def best_tprfpr(self, thresholds:np.ndarray, tprs: np.ndarray, fprs: np.ndarray) -> tuple:
        tpr = tprs[thresholds == self.threshold][0]
        fpr = fprs[thresholds == self.threshold][0]
        return (self.threshold, tpr, fpr)