
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils import One_vc_All, getTPRFPR

class ACC(Quantifier):
    """ Implementation of Adjusted Classify and Count
    """
    
    def __init__(self, classifier:BaseEstimator, threshold:float=0.5, random_state:int=None, n_jobs:int=1):
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        
        if random_state:
            classifier.set_params(**{"random_state":random_state, "n_jobs":n_jobs})
        
        self.__classifier = classifier
        self.__threshold = threshold
        self.__n_class = 2
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.classes = None
        self.one_vs_all = None
        self.tprfpr = None
    
    def fit(self, X, y):
        
        self.classes = np.unique(y)
        self.__n_class = len(np.unique(y))
        
        if self.__n_class > 2:
            self.one_vs_all = One_vc_All(X, y)
            
        self.__classifier.fit(X, y)
        
        scores = self.__classifier.predict_proba(X)[:, 1]
        scores = np.stack([scores, np.asarray(y)], axis=1)
        
        tprfpr = getTPRFPR(scores)
        self.tprfpr = tprfpr[tprfpr['threshold'] == self.__threshold]
        
        return self
        
    def estimate(self, X):
        
        prevalences = {}
        
        if self.__n_class > 2:
            for _class, (x, y) in self.one_vs_all.generate_trains():
                self.fit(x, y)
                prevalences[_class] = self.estimate(X)[1]
                self.__n_class = len(prevalences)
            return prevalences

        scores = self.__classifier.predict_proba(X)


        for i, _class in enumerate(self.classes):
            scores_class = scores[:, i]
            
            count = len(scores_class[scores_class >= self.__threshold])
            #Faster than using for loop below    
            cc_ouput = count/len(scores_class)   
            
            if (float(self.tprfpr['tpr'][0]) - float(self.tprfpr['fpr'][0])) == 0:
                prevalence = cc_ouput
            else:
                prevalence = (cc_ouput - float(self.tprfpr['fpr'][0]))/(float(self.tprfpr['tpr'][0]) - float(self.tprfpr['fpr'][0]))   #adjusted class proportion
            

            if prevalence <= 0:                           #clipping the output between [0,1]
                prevalence = 0
            elif prevalence >= 1:
                prevalence = 1
            else:
                prevalence = prevalence

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