
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils import getTPRFPR

class MS(Quantifier):
    """ Implementation of Median Sweep
    """
    
    def __init__(self, classifier:BaseEstimator, threshold:float=0.5):
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        
        self.__classifier = classifier
        self.__threshold = threshold
        self.__n_class = 2
        self.__classes = None
        self.tprfpr = []
        
    
    def _get_tprfpr(self, X, y, i:int, _class:int) -> list:
            
        scores = self.__classifier.predict_proba(X)[:, i]
        scores = np.stack([scores, np.asarray(y)], axis=1)
            
        tprfpr = getTPRFPR(scores, _class)
        
        return tprfpr
    
    
    def fit(self, X, y):
        
        self.__classes = np.unique(y)
        self.__n_class = len(np.unique(y))
        
        self.__classifier.fit(X, y)
        
        if self.__n_class > 2:
            for i in range(self.__n_class):
                self.tprfpr.append(self._get_tprfpr(X, y, i))
                
        
        self.tprfpr = self._get_tprfpr(X, y, 0, self.__classes[0])
            
            
        return self
    
    
    def _median_sweep(self, scores: np.ndarray, tprfpr) -> float:
        unique_scores  = np.arange(0.01,1,0.01)  #threshold values from 0.01 to 0.99  
        prevalances_array = []
        for i in unique_scores:        
            
            threshold, fpr, tpr = tprfpr[round(tprfpr['threshold'],4) == round(i,4)].to_numpy()[0] 
               
            class_prop = len(np.where(scores >= threshold)[0])/len(scores)

            if (tpr - fpr) == 0:
                prevalances_array.append(class_prop)           
            else:
                prevalence = (class_prop - fpr)/(tpr - fpr)        
                prevalances_array.append(prevalence)
                         
        prevalence = np.median(prevalances_array)   #adjusted class proportion
            
        prevalence = 1 if prevalence >= 1 else prevalence
        prevalence = 0 if prevalence <= 0 else prevalence
        
        return prevalence
        
    def predict(self, X) -> dict:
        
        prevalences = {}

        scores = self.__classifier.predict_proba(X)
        for i, _class in enumerate(self.__classes):
            scores_class = scores[:, i]
            
            if self.__n_class > 2:
                prevalence = self._median_sweep(scores_class, self.tprfpr[i])
                prevalences[_class] = np.round(prevalence, 3)
            else:

                if len(prevalences) > 0:
                    prevalences[_class] = np.round(1 - prevalences[self.__classes[0]], 3)
                    
                    return prevalences

                prevalence = self._median_sweep(scores_class, self.tprfpr) 

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