from typing import List
from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
import numpy as np
from joblib import Parallel, delayed
from copy import deepcopy
from functools import partial


class Quantifier(ABC, BaseEstimator):
    """ Abstract class for all types of quantifiers
    """
    
    @abstractmethod
    def fit(self, *args, **kwargs):
        ...
        
    @abstractmethod
    def predict(self, *args, **kwargs):
        ...



class AggregativeQuantifier(Quantifier):
    """Generic class for aggregative quantification, in this case, all aggregative methods use a classifier
    """
    
    
    def __init__(self):
        self.n_classes = None
        self.classes = None
        self.binary_quantifiers = None


    def fit(self, X, y):
        if len(np.unique(y)) > 2:

            self.binary_quantifiers = {_class: deepcopy(self) for _class in np.unique(y)}
            self._parallel(self._delayed_binary_fit, X, y)
            
            self.n_classes = len(np.unique(y))
        else:
            # Treina o quantificador diretamente para binário no método fit
            self.n_classes = 2
            self._fit_binary(X, y)
            
        return self


    def predict(self, X):
        if self.n_classes > 2:
            for i in range(self.n_classes):
                prevalences = self._parallel(self._delayed_binary_predict, X)
            summ = prevalences.sum(axis=-1, keepdims=True)
            prevalences = np.true_divide(prevalences, summ, where=summ>0)
            prevalences = {_class:prev for _class, prev in zip(self.classes, prevalences)}
        else:
            # Faz previsões para dados binários
            prevalences = self._predict_binary(X)

        return prevalences


    def _parallel(self, func, *args, **kwargs):
        return np.asarray(
            Parallel(n_jobs=-1, backend='threading')(
                delayed(func)(c, *args, **kwargs) for c in self.classes
            )
        )


    def _delayed_binary_predict(self, _class, X):
        return self.binary_quantifiers[_class].predict(X)[1]
    
    def _delayed_binary_fit(self, _class, X, y,):
        y_class = (y == _class).astype(int)
        return self.binary_quantifiers[_class].fit(X, y_class)
    
    
    
    @abstractmethod
    def _fit_binary(self, X, y):
        ...
        
    @abstractmethod 
    def _predict_binary(self, X):
        ...