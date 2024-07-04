
import numpy as np
from sklearn.base import BaseEstimator
import pdb

from ...base import Quantifier
from ...utils.utilities import get_values

class PCC(Quantifier):
    """ Implementation of Probabilistic Classify and Count
    """
    
    def __init__(self, classifier:BaseEstimator, round_to:int=3):
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        
        self.classifier = classifier
        self.round_to = round_to
        self.tprfpr = None
    
    def _fit_binary(self, X, y):
        self.classifier.fit(X, y)
        return self
    
    def _fit_multiclass(self, X, y):
        self.classifier.fit(X, y)
        return self
    
    def _predict_multiclass(self, X):
        prevalences = {}
        
        scores = self.classifier.predict_proba(X)
        
        for i, _class in enumerate(self.classes):
            prevalences[_class] = np.round(np.mean(scores[:, i]), self.round_to)
        
        return prevalences
    
        
    def _predict_binary(self, X) -> dict:
        
        prevalences = {}
        
        scores = self.classifier.predict_proba(X)
        
        prevalence =  np.mean(scores[:, 1])
        prevalences[self.classes[0]] = np.round(1 - prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(prevalence, self.round_to)
        
        return prevalences
    