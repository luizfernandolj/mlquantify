
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils.utilities import get_values

class MAX(Quantifier):
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
        diff_tpr_fpr = list(abs(tprfpr['tpr'] - tprfpr['fpr']))
    
        max_index = diff_tpr_fpr.index(max(diff_tpr_fpr))         #Finding index where (tpr-fpr) is maximum
        threshold, tpr, fpr = tprfpr.loc[max_index]
        
        self.tprfpr = [threshold, tpr, fpr]
            
        return self
    
        
    def _predict_binary(self, X) -> dict:
        
        prevalences = {}
        
        scores = self.classifier.predict_proba(X)
        
        scores_class = scores[:, 1]
        
        threshold, tpr, fpr = self.tprfpr
        prevalence = self._threshold_max(scores_class, threshold, tpr, fpr)
        
        prevalences[self.classes[0]] = np.round(1 - prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(prevalence, self.round_to)
        
        return prevalences
    
        
    def _threshold_max(self, scores: np.ndarray, threshold:float, tpr:float, fpr:float) -> float:
        class_prop = len(np.where(scores >= threshold)[0])/len(scores)
            
        if (tpr - fpr) == 0:
            prevalence = class_prop
        else:
            prevalence = (class_prop - fpr)/(tpr - fpr)
        
        return np.clip(prevalence, 0, 1)