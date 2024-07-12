from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from .utils.utilities import parallel, normalize_prevalence

class Quantifier(ABC, BaseEstimator):
    """ Abstract Class for quantifiers """
    
    @abstractmethod
    def fit(self, X, y) -> object: ...
    
    @abstractmethod
    def predict(self, X) -> dict: ...
    
    @property
    def classes(self) -> list:
        return self._classes
    
    @classes.setter
    def classes(self, classes):
        self._classes = classes
    
    @property
    def n_class(self) -> list:
        return len(self._classes)
    
    @property
    def multiclass_method(self) -> bool:
        return True
    
    @property
    def probabilistic_method(self) -> bool:
        return True

    @property
    def binary_data(self) -> bool:
        return len(self._classes) == 2
    
    
    
        

class AggregativeQuantifier(Quantifier, ABC):
    
    
    def __init__(self):
        self.ova = None
        self.binary_quantifiers = {}
    

    def fit(self, X, y, learner_fitted=False):
        self.classes = np.unique(y)
        #Binary quantification or multiclass quantification if method is multiclass it self
        if self.binary_data or self.multiclass_method:  
            return self._fit_method(X, y, learner_fitted)
        
        # Making one vs all
        qtf_fitted = parallel(self.delayed_fit, self.classes, X, y, learner_fitted)
        self.binary_quantifiers = {class_:qtf for class_, qtf in zip(self.classes, qtf_fitted)}
        
        return self
        

    def predict(self, X) -> dict:
        
        if self.binary_data or self.multiclass_method:
            prevalences = self._predict_method(X)
            return normalize_prevalence(prevalences)
        # Making one vs all
        prevalences = parallel(self.delayed_predict, self.classes, X)
        return normalize_prevalence(prevalences)
    
    @abstractmethod
    def _fit_method(self, X, y): ...
    
    @abstractmethod
    def _predict_method(self, X) -> dict: ...
    

    @property
    def learner(self):
        return self.learner_

    @learner.setter
    def learner(self, value):
        self.learner_ = value
        
        
    # MULTICLASS METHODS
    
    def delayed_fit(self, class_, X, y, learner_fitted):
        return self.binary_quantifiers[class_]._fit_method(X, y, learner_fitted)
    
    def delayed_predict(self, class_, X):
        return self.binary_quantifiers[class_]._predict_method(X)
    
    
    @classmethod
    def GetScores(self, X, y, folds:int=10, learner_fitted:bool=False) -> tuple:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
            
        if learner_fitted:
            probabilities = self.learner.predict_proba(X)[:, 1]
            y_label = y
        else:
        
            skf = StratifiedKFold(n_splits=folds)    
            probabilities = []
            y_label = []
            
            for train_index, valid_index in skf.split(X,y):
                
                tr_data = pd.DataFrame(X.iloc[train_index])   #Train data and labels
                tr_lbl = y.iloc[train_index]
                
                valid_data = pd.DataFrame(X.iloc[valid_index])  #Validation data and labels
                valid_lbl = y.iloc[valid_index]
                
                self.learner.fit(tr_data, tr_lbl)
                
                probabilities.extend(self.learner.predict_proba(valid_data)[:,1])     #evaluating scores
                y_label.extend(valid_lbl)
        
        return y, probabilities
    
    
    
    
    
class ThresholdOptimization(AggregativeQuantifier):
    
    
    def __init__(self, learner: BaseEstimator, threshold:float=0.5):
        self.learner = learner
        self.threshold = threshold
        self.cc_output = None
        self.tpr = None
        self.fpr = None
    
    
    def multiclass_method(self):
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
        
            skf = StratifiedKFold(n_splits=cv_folds)    
            probabilities = []
            y_label = []
            
            for train_index, valid_index in skf.split(X,y):
                
                tr_data = pd.DataFrame(X.iloc[train_index])   #Train data and labels
                tr_lbl = y.iloc[train_index]
                
                valid_data = pd.DataFrame(X.iloc[valid_index])  #Validation data and labels
                valid_lbl = y.iloc[valid_index]
                
                self.learner.fit(tr_data, tr_lbl)
                
                probabilities.extend(self.learner.predict_proba(valid_data)[:,1])     #evaluating scores
                y_label.extend(valid_lbl)
        
        probabilities = np.asarray(probabilities)
        #y_labels, probabilities = self.GetScores(X, y, folds=cv_folds, learner_fitted=learner_fitted)
        
        self.learner.fit(X, y)
        
        self.cc_output = len(probabilities[probabilities >= self.threshold]) / len(probabilities)
        
        thresholds, tprs, fprs = self.adjust_threshold(y_label, probabilities)
        self.tpr, self.fpr = self.best_tprfpr(thresholds, tprs, fprs)
        
        return self
    
    
    def _predict_method(self, X):
        prevalences = {}
        
        if self.tpr - self.fpr == 0:
            return self.cc_output
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
        unique_scores = np.linspace(0,1,100)
        
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
    
    
            
            