
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils.utilities import get_values

class ACC(Quantifier):
    """ Implementation of Adjusted Classify and Count
    """
    
    def __init__(self, classifier:BaseEstimator, threshold:float=0.5, round_to:int=3):
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        
        self.classifier = classifier
        self.threshold = threshold
        self.round_to = round_to
        self.tprfpr = None
    
    def _fit_binary(self, X, y):
        self.classifier.fit(X, y)
        
        values = get_values(X, y, self.classifier, tprfpr=True)
        tprfpr = values["tprfpr"]
        #getting tpr and fpr for threshold equals threshold argument of the class
        threshold, tpr, fpr = tprfpr[tprfpr['threshold'] == self.threshold].to_numpy()[0]
        
        self.tprfpr = [threshold, tpr, fpr]
            
        return self
        
    def _predict_binary(self, X) -> dict:
        
        prevalences = {}
        
        scores = self.classifier.predict_proba(X)
        
        scores_class = scores[:, 1]
        _, tpr, fpr = self.tprfpr
        prevalence = self._adjust_classify_count(scores_class, tpr, fpr)
        prevalences[self.classes[0]] = np.round(prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(1 - prevalence, self.round_to)
        
        return prevalences
    
        
    def _adjust_classify_count(self, scores: np.ndarray, tpr:float, fpr:float) -> float:
        count = len(scores[scores >= self.threshold])
        cc_output = count / len(scores)
        
        prevalence = (cc_output - fpr) / (tpr - fpr) if tpr != fpr else cc_output
        
        return np.clip(prevalence, 0, 1)
    
    