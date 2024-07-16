from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import StratifiedKFold

from .utils.utilities import parallel, normalize_prevalence, GetScores

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
        self.binary_quantifiers = {class_:deepcopy(self) for class_ in self.classes}
        parallel(self.delayed_fit, self.classes, X, y, learner_fitted)
        
        
        return self
        

    def predict(self, X) -> dict:
        
        if self.binary_data or self.multiclass_method:
            prevalences = self._predict_method(X)
            return normalize_prevalence(prevalences, self.classes)
        # Making one vs all 
        prevalences = parallel(self.delayed_predict, self.classes, X)
        #print(prevalences)
        return normalize_prevalence(prevalences, self.classes)
    
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
        y_class = (y == class_).astype(int)
        return self.binary_quantifiers[class_]._fit_method(X, y_class, learner_fitted)
    
    def delayed_predict(self, class_, X):
        return self.binary_quantifiers[class_]._predict_method(X)[1]
    
    
    
    
    
    
