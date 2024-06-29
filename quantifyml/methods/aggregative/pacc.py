import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils import getTPRFPR

class PACC(Quantifier):
    
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
        threshold, tpr, fpr = tprfpr[tprfpr['threshold'] == self.__threshold].to_numpy()[0]
        
        return [threshold, tpr, fpr]
    
    def fit(self, X, y):
        
        self.__classes = np.unique(y)
        self.__n_class = len(np.unique(y))
        
        self.__classifier.fit(X, y)
        
        if self.__n_class > 2:
            for i in range(self.__n_class):
                self.tprfpr.append(self._get_tprfpr(X, y, i))
                
        
        self.tprfpr = self._get_tprfpr(X, y, 0, self.__classes[0])
            
            
        return self
    
    
    def _adjust_pcc(self, scores_mean, tpr, fpr) -> float:
        diff_tpr_fpr = tpr - fpr
        prevalence = (scores_mean - fpr) / diff_tpr_fpr  #adjusted class proportion
        
        prevalence = 1 if prevalence >= 1 else prevalence
        prevalence = 0 if prevalence <= 0 else prevalence
        
        return prevalence

        
    
        
    def predict(self, X):
        
        prevalences = {}

        scores = self.__classifier.predict_proba(X)


        for i, _class in enumerate(self.__classes):
            scores_class = scores[:, i]
            scores_class_mean = np.mean(scores_class)
            
            if self.__n_class > 2:
                _, tpr, fpr = self.tprfpr[i]
                prevalence = self._adjust_classify_count(scores_class, tpr, fpr)
                prevalences[_class] = np.round(prevalence, 3)
            else:  
                if len(prevalences) > 0:
                    prevalences[_class] = 1 - prevalences[self.__classes[0]]
                    
                    return prevalences
                
                _, tpr, fpr = self.tprfpr
            
                prevalence = self._adjust_pcc(scores_class_mean, tpr, fpr)

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