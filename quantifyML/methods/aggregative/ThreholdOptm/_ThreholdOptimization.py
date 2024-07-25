from abc import abstractmethod
import numpy as np
from sklearn.base import BaseEstimator

from ....base import AggregativeQuantifier
from ....utils import adjust_threshold, get_scores

class ThresholdOptimization(AggregativeQuantifier):
    # Class for optimizing classification thresholds

    def __init__(self, learner: BaseEstimator):
        self.learner = learner
        self.threshold = None
        self.cc_output = None
        self.tpr = None
        self.fpr = None
    
    @property
    def multiclass_method(self) -> bool:
        return False
    
    def _fit_method(self, X, y):

        y_labels, probabilities = get_scores(X, y, self.learner, self.cv_folds, self.learner_fitted)
        
        # Adjust thresholds and compute true and false positive rates
        thresholds, tprs, fprs = adjust_threshold(y_labels, probabilities[:, 1], self.classes)
        
        # Find the best threshold based on TPR and FPR
        self.threshold, self.tpr, self.fpr = self.best_tprfpr(thresholds, tprs, fprs)
        
        return self
    
    def _predict_method(self, X) -> dict:
              
        probabilities = self.learner.predict_proba(X)[:, 1]
        
        # Compute the classification count output
        self.cc_output = len(probabilities[probabilities >= self.threshold]) / len(probabilities)
        
        # Calculate prevalence, ensuring it's within [0, 1]
        if self.tpr - self.fpr == 0:
            prevalence = self.cc_output
        else:
            prevalence = np.clip((self.cc_output - self.fpr) / (self.tpr - self.fpr), 0, 1)
        
        prevalences = [1- prevalence, prevalence]

        return prevalences
    
    @abstractmethod
    def best_tprfpr(self, thresholds: np.ndarray, tpr: np.ndarray, fpr: np.ndarray) -> float:
        # Abstract method for determining the best threshold based on TPR and FPR
        ...
