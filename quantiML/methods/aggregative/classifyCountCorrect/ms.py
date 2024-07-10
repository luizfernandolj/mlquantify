
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils.utilities import get_values

class MS(Quantifier):
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
        self.tprfpr = values["tprfpr"]
            
        return self
        
    def _predict_binary(self, X) -> dict:
        
        prevalences = {}
        
        scores = self.classifier.predict_proba(X)
        
        scores_class = scores[:, 1]
        prevalence = self._median_sweep(scores_class, self.tprfpr)
        prevalences[self.classes[0]] = np.round(1 - prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(prevalence, self.round_to)
        
        return prevalences
        
        
    def _median_sweep(self, scores: np.ndarray, tprfpr) -> float:
        unique_scores  = np.arange(0.01,1,0.01)  #threshold values from 0.01 to 0.99  
        prevalances_array = []
        for i in unique_scores:        
            
            threshold, tpr, fpr = tprfpr[round(tprfpr['threshold'],4) == round(i,4)].to_numpy()[0] 
               
            class_prop = len(np.where(scores >= threshold)[0])/len(scores)

            if (tpr - fpr) == 0:
                prevalances_array.append(class_prop)           
            else:
                prevalence = (class_prop - fpr)/(tpr - fpr)        
                prevalances_array.append(prevalence)
                         
        prevalence = np.median(prevalances_array)
        
        return np.clip(prevalence, 0, 1) 