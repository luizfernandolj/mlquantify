import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from ...base import Quantifier
from ...utils import One_vs_All, getHist, get_distance, ternary_search, get_scores

class HDy(Quantifier):
    """ Implementation of HDy
    """    
    
    def __init__(self, classifier:BaseEstimator):
        
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        
        self.__classifier = classifier
        self.__n_class = 2
        self.__classes = None
        self.__pos_neg_scores = {}
        self.one_vs_all = None
        self.__distance = {}
    
        
    def fit(self, X, y):
        self.__classes = np.unique(y)
        self.__n_class = len(np.unique(y))
        
        return self
        
        
    def _mixture_hellinger_distance(self, pos_scores, neg_scores, test_scores):
        bin_size = np.linspace(2,20,10)   #creating bins from 2 to 10 with step size 2
        bin_size = np.append(bin_size, 30)
        
        result  = []
        for bins in bin_size:
            #....Creating Histograms bins score\counts for validation and test set...............
            
            p_bin_count = getHist(pos_scores, bins)
            n_bin_count = getHist(neg_scores, bins)
            te_bin_count = getHist(test_scores, bins)
            
            def f(x):            
                return(get_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure = "hellinger"))
        
            result.append(ternary_search(0, 1, f))                                           
                            
        prevalence = np.median(result)
        
        return prevalence
        
    
        
    def predict(self, X) -> dict:
        
        prevalences = {}

        scores = self.__classifier.predict_proba(X)
        
        for i, _class in enumerate(self.__classes):
            scores_class = scores[:, ]
            
            if self.__n_class > 2:
                pos_scores, neg_scores = self.__pos_neg_scores[_class]
                prevalence = self._mixture_hellinger_distance(pos_scores, neg_scores, scores_class)
                prevalences[_class] = np.round(prevalence, 3)
            else:

                if len(prevalences) > 0:
                    prevalences[_class] = np.round(1 - prevalences[self.__classes[0]], 3)
                    
                    return prevalences

                pos_scores, neg_scores = self.__pos_neg_scores
            
                prevalence = self._mixture_hellinger_distance(pos_scores, neg_scores, scores_class)

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