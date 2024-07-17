
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils.utilities import get_values

class PACC(Quantifier):
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
        
        scores_mean = np.mean(scores[:, 1])
        _, tpr, fpr = self.tprfpr
        prevalence = self._probabilistic_adjust_classify_count(scores_mean, tpr, fpr)
        prevalences[self.classes[0]] = np.round(1 - prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(prevalence, self.round_to)
        
        return prevalences
    
        
    def _probabilistic_adjust_classify_count(self, mean_scores: float, tpr:float, fpr:float) -> float:
        diff_tpr_fpr = tpr - fpr
        
        print(mean_scores, fpr, tpr)
        prevalence = (mean_scores - fpr) / diff_tpr_fpr if diff_tpr_fpr != 0 else mean_scores
        
        return np.clip(prevalence, 0, 1)
    
    