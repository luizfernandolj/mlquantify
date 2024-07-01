
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier, Utils
from ...utils import getTPRFPR

class ACC(Quantifier, Utils):
    """ Implementation of Adjusted Classify and Count
    """
    
    def __init__(self, classifier:BaseEstimator, threshold:float=0.5, round_to:int=3):
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        
        self.classifier = classifier
        self.threshold = threshold
        self.n_class = 2
        self.classes = None
        self.round_to = round_to
        self.tprfpr = []
    
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        self.n_class = len(np.unique(y))
        self.classifier.fit(X, y)
        
        if self.n_class > 2 or not self.is_binary(y):
             # Applying one vs all for each class if number of class is greater than 2
            for _, y_class in self.one_vs_all(y):
                values = self.get_values(X, y_class, self.classifier, tprfpr=True)
                tprfpr = values["tprfpr"]
                
                threshold, tpr, fpr = tprfpr[tprfpr['threshold'] == self.threshold].to_numpy()[0]
                
                self.tprfpr.append([threshold, tpr, fpr])
            return self
        
        values = self.get_values(X, y, self.classifier, tprfpr=True)
        tprfpr = values["tprfpr"]
        
        #getting tpr and fpr for threshold equals threshold argument of the class
        threshold, tpr, fpr = tprfpr[tprfpr['threshold'] == self.threshold].to_numpy()[0]
        
        self.tprfpr = [threshold, tpr, fpr]
            
        return self
    
    
    def _adjust_classify_count(self, scores: np.ndarray, tpr:float, fpr:float) -> float:
        count = len(scores[scores >= self.threshold])
        cc_output = count / len(scores)
        
        prevalence = (cc_output - fpr) / (tpr - fpr) if tpr != fpr else cc_output
        
        return np.clip(prevalence, 0, 1)
    
        
    def predict(self, X) -> dict:
        
        prevalences = {}
        
        scores = self.classifier.predict_proba(X)

        if self.n_class > 2 or not self.binary: 
            
            for i, (_class, [_, tpr, fpr]) in enumerate(zip(self.classes, self.tprfpr)):
                scores_class = scores[:, i]
                prevalence = self._adjust_classify_count(scores_class, tpr, fpr)
                prevalences[_class] = np.round(prevalence, self.round_to)
            prevalences = {_class:round(p/sum(prevalences.values()), self.round_to) for _class, p in prevalences.items()}
            return prevalences
        
        scores_class = scores[:, 1]
        _, tpr, fpr = self.tprfpr
        prevalence = self._adjust_classify_count(scores_class, tpr, fpr)
        prevalences[1] = np.round(prevalence, self.round_to)
        prevalences[0] = np.round(1 - prevalence, self.round_to)
        
        return prevalences
    