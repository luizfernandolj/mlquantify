
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ....base import AggregativeQuantifier
from ....utils import adjust_threshold, GetScores

class ThresholdOptimization(AggregativeQuantifier):
    
    
    def __init__(self, learner: BaseEstimator):
        self.learner = learner
        self.threshold = None
        self.cc_output = None
        self.tpr = None
        self.fpr = None
    
    @property
    def multiclass_method(self) -> bool:
        return False
    
    
    def _fit_method(self, X, y, learner_fitted:bool=False, cv_folds:int=10):
        
        y_label, probabilities = GetScores(X, y, self.learner, cv_folds, learner_fitted)
        self.learner.fit(X, y) if learner_fitted is False else None
        
        thresholds, tprs, fprs = adjust_threshold(y_label, probabilities[:, 1], self.classes)
        
        self.threshold, self.tpr, self.fpr = self.best_tprfpr(thresholds, tprs, fprs)
        
        return self
    
    
    def _predict_method(self, X):
        prevalences = {}
        
        probabilities = self.learner.predict_proba(X)[:, 1]
        
        self.cc_output = len(probabilities[probabilities >= self.threshold]) / len(probabilities)
        
        if self.tpr - self.fpr == 0:
            prevalence = self.cc_output
        else:
            prevalence = np.clip((self.cc_output - self.fpr) / (self.tpr - self.fpr), 0, 1)
        
        prevalences[self.classes[1]] = prevalence
        prevalences[self.classes[0]] = 1 - prevalence

        return prevalences
    
    
    @abstractmethod
    def best_tprfpr(self, thresholds:np.ndarray, tpr:np.ndarray, fpr:np.ndarray) -> float:
        ...
    
    
            
            