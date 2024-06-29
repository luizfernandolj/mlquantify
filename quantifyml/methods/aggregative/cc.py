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
        self.__n_class = 2
        self.__classes = None
    
    
    def fit(self, X, y):
        self.__classes = np.unique(y)
        self.__n_class = len(np.unique(y))
             
        self.__classifier.fit(X, y)
        
        return self
        
        
    def predict(self, X) -> dict:
        
        y_pred = self.__classifier.predict(X)
        classes, nclasses = np.unique(y_pred, return_counts=True)
        
        return { _class : round(nclass/len(y_pred), 3) for _class, nclass in zip(classes, nclasses) }
    
    
    @property
    def n_class(self):
        return self.__n_class
    
    @property
    def classifier(self):
        return self.__classifier
    
    @classifier.setter
    def classifier(self, new_classifier):
        assert isinstance(new_classifier, BaseEstimator), "Classifier object is not an estimator"
        
        self.__classifier = new_classifier