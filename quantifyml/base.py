from typing import List
from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
import numpy as np
from copy import deepcopy
from .utils.utilities import parallel, normalize_prevalence

class Quantifier(ABC):
    """ Class that performs fit and predict dynamically for binary and multiclass quantification """
    
    
    def __init__(self):
        self.n_class = None
        self.classes = None
        self.binary_quantifiers = {}


    def fit(self, X, y):
        """ Fits the quantifier to the data """
        self.is_binary = self._detect_binary(y)

        if self.is_binary:
            self._fit_binary(X, y) # Binary quantification
        else:
            self._fit_multiclass(X, y) # Multiclass quantification
        return self

    def predict(self, X):
        """ Predicts the prevalences for new data """
        if self.is_binary:
            prevalences = self._predict_binary(X)
        else:
            prevalences = self._predict_multiclass(X)
            if isinstance(prevalences, dict):
                return prevalences
            if len(prevalences) == 3:
                return prevalences
            prevalences = normalize_prevalence(prevalences, self.classes)

        return prevalences

    def _detect_binary(self, y):
        """ Detects if the task is binary """
        self.classes = np.unique(y)
        self.n_class = np.unique(y).shape[0]
        return self.n_class == 2



    @abstractmethod
    def _fit_binary(self, X, y):
        """abstract method to perform fit for binary quantification """
        ...
        
    @abstractmethod 
    def _predict_binary(self, X):
        """ Performs prediction for binary quantification """
        ...
    
    def _fit_multiclass(self, X, y):
        """ Fits multiple binary quantifiers (One-vs-All), but other multiclass methods can overwrite it"""
            
        quantifiers_fitted = parallel(self._delayed_binary_fit, self.classes, X, y)   # paralell fitting
        # ================================================================
        self.binary_quantifiers = {_class: qtf for _class, qtf in zip(np.unique(y), quantifiers_fitted)}
        
    
    def _predict_multiclass(self, X):
        """ Predicts using multiple binary quantifiers (One-vs-All) but other multiclass methods can overwrite it"""
        
        predictions = parallel(self._delayed_binary_predict,self.classes, X)   #parallel predicting
        return predictions
    
    
    
    def _delayed_binary_fit(self, _class, X, y,):
        """ Function for parallel binary fit """
        qtf = deepcopy(self)
        y_class = (y == _class).astype(int)
        return qtf._fit_binary(X, y_class)
    
    def _delayed_binary_predict(self, _class, X):
        """ Function for parallel binary prediction """
        return self.binary_quantifiers[_class]._predict_binary(X)[1]