
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils.utilities import get_values

class MS2(Quantifier):
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
        prevalence = self._median_sweep2(scores_class, self.tprfpr)
        prevalences[self.classes[0]] = np.round(1 - prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(prevalence, self.round_to)
        
        return prevalences
        
        
    def _median_sweep2(self, scores: np.ndarray, tprfpr) -> float:
        index = np.where(abs(tprfpr['tpr'] - tprfpr['fpr']) >(1/4) )[0].tolist()
        if index == 0:
            index = np.where(abs(tprfpr['tpr'] - tprfpr['fpr']) >=0 )[0].tolist()

        
        prevalances_array = []    
        for i in index:
            
            threshold, tpr, fpr = tprfpr.loc[i]
            estimated_positive_ratio = len(np.where(scores >= threshold)[0])/len(scores)
            
            diff_tpr_fpr = abs(float(tpr-fpr))  
        
            if diff_tpr_fpr == 0.0:            
                diff_tpr_fpr = 1     
        
            final_prevalence = abs(estimated_positive_ratio - fpr)/diff_tpr_fpr
            
            prevalances_array.append(final_prevalence)  
    
        prevalence = np.median(prevalances_array)
            
        return np.clip(prevalence, 0, 1) 