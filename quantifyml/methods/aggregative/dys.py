
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils.utilities import get_values
from ...utils import getHist, get_distance, ternary_search

class DyS(Quantifier):
    """ Implementation of Adjusted Classify and Count
    """
    
    def __init__(self, classifier:BaseEstimator, round_to:int=3, measure:str="topsoe", return_distance:str="none"):
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        assert return_distance in ["none", "only"], "return distance is not a valid choice"
        
        self.classifier = classifier
        self.round_to = round_to
        self.measure = measure
        self.return_distance = return_distance
        self.pos_scores = None
        self.neg_scores = None
    
    def _fit_binary(self, X, y):
        self.classifier.fit(X, y)
        
        _class = np.unique(y)[1]
        
        values = get_values(X, y, self.classifier, scores=True)
        scores = values["scores"]
        
        self.pos_scores = scores[scores[:, 1] == _class][:, 0]
        self.neg_scores = scores[scores[:, 1] != _class][:, 0]
            
        return self
        
    def _predict_binary(self, X) -> dict:
        
        prevalences = {}
        
        scores = self.classifier.predict_proba(X)[:, 1]
        
        if self.return_distance == "only":
            distance = self._dys_distance(scores)
            return {self.classes[0]:distance, self.classes[1]:distance, "a":1}
        
        prevalence = self._dys_distance(scores)
        prevalences[self.classes[0]] = np.round(1 - prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(prevalence, self.round_to)
        
        return prevalences
    
        
    def _dys_distance(self, scores: np.ndarray) -> float:
        bin_size = np.linspace(2,20,10)   #creating bins from 2 to 10 with step size 2
        bin_size = np.append(bin_size, 30)
        
        result  = []
        for bins in bin_size:
            
            p_bin_count = getHist(self.pos_scores, bins)
            n_bin_count = getHist(self.neg_scores, bins)
            te_bin_count = getHist(scores, bins) 
            
            def f(x):            
                return(get_distance(((p_bin_count*x) + (n_bin_count*(1-x))), te_bin_count, measure=self.measure))
        
            result.append(ternary_search(0, 1, f))                                           
                            
        prevalence = np.median(result)
        
        if self.return_distance == "only":
            index = np.where(sorted(result) == prevalence)[0].astype(int)[0]
            
            p_bin_count = getHist(self.pos_scores, bin_size[index])
            n_bin_count = getHist(self.neg_scores, bin_size[index])
            te_bin_count = getHist(scores, bin_size[index]) 
            
            distance = get_distance(((p_bin_count*prevalence) + (n_bin_count*(1-prevalence))), te_bin_count, measure = self.measure)
            
            return np.round(distance, self.round_to)
        
        return np.clip(prevalence, 0, 1)
        
    
    