
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier

class PCC(Quantifier):
    
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
        
    def predict(self, X):
        
        prevalences = {}
        
        scores = self.__classifier.predict_proba(X)
        
        for i, _class in enumerate(self.__classes):
            
            if self.__n_class > 2:
                prevalences[_class] = np.round(np.mean(scores[:, i]), 3)
            else:             
                if len(prevalences) > 0:
                    prevalences[_class] = 1 - prevalences[self.__classes[0]]
                    
                    return prevalences
            
                prevalence = np.round(np.mean(scores[:, i]), 3)  

                prevalences[_class] = np.round(prevalence, 3)    
            
        
        return prevalences
        
    
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