from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
import numpy as np
from utils import normalize

class Quantifier(ABC, BaseEstimator):
    """ Abstract Class for quantifiers """
    
    @abstractmethod
    def fit(self, X, y) -> object: ...
    
    @abstractmethod
    def predict(self, X) -> dict: ...
    
    @property
    @abstractmethod
    def classes(self) -> list: ...
    
    @property
    def n_class(self) -> list:
        return len(self._classes)
    
    @property
    def multiclass_method(self) -> bool:
        return True

    @property
    def binary_data(self) -> bool:
        return len(self._classes) == 2
    
    
    
        

class AggregativeQuantifier(Quantifier, ABC):
    
    
    def __init__(self):
        self.__ova = None 


    def fit(self, X, y, learner_fitted=False):

        #Binary quantification or multiclass quantification if method is multiclass it self
        if self.binary_data or self.multiclass_method:  
            return self._fit_method(X, y, learner_fitted)
        
        # Making one vs all
        self.__ova = OneVsAll(self, self._fit_method, self.classes)
        return self.__ova.fit(X, y, learner_fitted)
        

    def predict(self, X) -> dict:
        
        if self.binary_data or self.multiclass_method:
            prevalences = self._predict_method(X)
            return normalize(prevalences)
        
        prevalences = self.__ova.predict(X)
        return normalize(prevalences)

    


     @property
    def learner(self):
        return self.learner_

    @learner.setter
    def learner(self, value):
        self.learner_ = value
        
    @property
    def classes(self) -> list:
        return self.learner.classes_
        


class OneVsAll:
    
    def fit(self): ...
    
    def predict(self): ...