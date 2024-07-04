
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils.utilities import get_values
from ...utils import getHist, get_distance

class HDy(Quantifier):
    """ Implementation of Adjusted Classify and Count
    """
    
    def __init__(self, classifier:BaseEstimator, round_to:int=3, return_distance:str="none"):
        assert isinstance(classifier, BaseEstimator), "Classifier object is not an estimator"
        assert return_distance in ["none", "only", "all"], "return distance is not a valid choice"
        
        self.classifier = classifier
        self.round_to = round_to
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
        
        prevalence = self._hellinger_distance(scores)
        prevalences[self.classes[0]] = np.round(1 - prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(prevalence, self.round_to)
        
        return prevalences
    
        
    def _hellinger_distance(self, scores: np.ndarray) -> float:
        bin_size = np.linspace(10,110,11)       #creating bins from 10 to 110 with step size 10
    #alpha_values = [round(x, 2) for x in np.linspace(0,1,101)]
        alpha_values = np.linspace(0,1,101)
        
        result = []
 
        for bins in bin_size:
            
            p_bin_density = getHist(self.pos_scores, bins)
            n_bin_density = getHist(self.neg_scores, bins)
            te_bin_density = getHist(scores, bins) 

            vDist = []
            
            for x in alpha_values:
                x = np.round(x,2)
                vDist.append(get_distance(((p_bin_density*x) + (n_bin_density*(1-x))), te_bin_density, measure="hellinger"))

            result.append(alpha_values[np.argmin(vDist)])
        
        
        prevalence = np.median(result)
            
        return np.clip(prevalence, 0, 1)
        
    
    