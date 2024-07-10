from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from utils.utilities import parallel, normalize_prevalence

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
    

    def fit(self, X, y, learner_fitted=False, *args):
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
    
    
    
    
class ClassifyCountCorrect(AggregativeQuantifier, ABC):
    
    
    
    @classmethod
    def GetScores(self, X, y, folds:int=10) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        
        skf = StratifiedKFold(n_splits=folds)    
        results = []
        class_labels = []
        
        for train_index, valid_index in skf.split(X,y):
            
            tr_data = pd.DataFrame(X.iloc[train_index])   #Train data and labels
            tr_lbl = y.iloc[train_index]
            
            valid_data = pd.DataFrame(X.iloc[valid_index])  #Validation data and labels
            valid_lbl = y.iloc[valid_index]
            
            self.learner.fit(tr_data, tr_lbl)
            
            results.extend(self.learner.predict_proba(valid_data)[:,1])     #evaluating scores
            class_labels.extend(valid_lbl)
        
        scores = np.c_[results,class_labels]
        
        return scores
    
    
    
        
    
    
    