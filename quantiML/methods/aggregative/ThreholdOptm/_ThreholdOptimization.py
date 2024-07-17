
from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ....base import AggregativeQuantifier
from ....utils.utilities import GetScores

class ThresholdOptimization(AggregativeQuantifier):
    
    
    def __init__(self, learner: BaseEstimator, threshold:float=0.5):
        self.learner = learner
        self.threshold = threshold
        self.cc_output = None
        self.tpr = None
        self.fpr = None
    
    @property
    def multiclass_method(self) -> bool:
        return False
    
    
    def _fit_method(self, X, y, learner_fitted:bool=False, cv_folds:int=10):
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
            
        if learner_fitted:
            probabilities = self.learner.predict_proba(X)[:, 1]
            y_label = y
        else:   
            y_label, probabilities = GetScores(X, y, self.learner, cv_folds, learner_fitted)
        
        
        probabilities = np.asarray(probabilities)
        
        self.learner.fit(X, y)
        
        thresholds, tprs, fprs = self.adjust_threshold(y_label, probabilities)
        
        self.tpr, self.fpr = self.best_tprfpr(thresholds, tprs, fprs)
        
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
    
    
    def count_predictions(self, y, y_pred) -> list:
        TP = np.logical_and(y == y_pred, y == self.classes[1]).sum()
        FP = np.logical_and(y != y_pred, y == self.classes[0]).sum()
        FN = np.logical_and(y != y_pred, y == self.classes[1]).sum()
        TN = np.logical_and(y == y_pred, y == self.classes[0]).sum()
        return TP, FP, TN, FN
    
    
    def get_tpr(self, TP, FP):
        if TP + FP == 0:
            return 0
        return TP / (TP + FP)

    def get_fpr(self, TN, FP):
        if FP + TN == 0:
            return 0
        return FP / (FP + TN)


    def adjust_threshold(self, y, probabilities:np.ndarray) -> tuple:
        unique_scores = np.linspace(0,1,101)
        
        tprs = []
        fprs = []
        
        for threshold in unique_scores:
            y_pred = np.where(probabilities < threshold, 0, 1)
            
            TP, FP, TN, _ = self.count_predictions(y, y_pred)
            
            tpr = self.get_tpr(TP, FP)
            fpr = self.get_fpr(TN, FP)   
            
            tprs.append(tpr)
            fprs.append(fpr)
        
        #best_tpr, best_fpr = self.adjust_threshold(np.asarray(tprs), np.asarray(fprs))
        return (unique_scores, np.asarray(tprs), np.asarray(fprs))
    
    
    @abstractmethod
    def best_tprfpr(self, thresholds:np.ndarray, tpr:np.ndarray, fpr:np.ndarray) -> float:
        ...
    
    
            
            