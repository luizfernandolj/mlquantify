import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier

from ...base import Quantifier

class CC(Quantifier):
    """ Implementation of Classify and Count quantification method
    """    
    
    def __init__(self, classifier:BaseEstimator):
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        
        self.__classifier = classifier
    
    
    def _fit_binary(self, X, y):
             
        self.__classifier.fit(X, y)
        
        return self
        
        
    def _predict_binary(self, X) -> dict:
        
        y_pred = self.__classifier.predict(X)
        classes, nclasses = np.unique(y_pred, return_counts=True)
        
        return { _class : round(nclass/len(y_pred), 3) for _class, nclass in zip(classes, nclasses) }
    
    
    def _fit_multiclass(self, X, y):
        self._fit_binary(X, y)
    
    def _predict_multiclass(self, X):
        return self._predict_binary(X)