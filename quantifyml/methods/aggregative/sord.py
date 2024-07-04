
import numpy as np
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ...utils.utilities import get_values
from ...utils import getHist, get_distance, ternary_search

class SORD(Quantifier):
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
        
        prevalence = self._sord(scores)
        prevalences[self.classes[0]] = np.round(1 - prevalence, self.round_to)
        prevalences[self.classes[1]] = np.round(prevalence, self.round_to)
        
        return prevalences
    
        
    def _sord(self, scores: np.ndarray) -> float:
        alpha = np.linspace(0,1,101)
        sc_1  = self.pos_scores
        sc_2  = self.neg_scores
        ts    = scores
        
        vDist   = []
        for k in alpha:        
            pos = np.array(sc_1)
            neg = np.array(sc_2)
            test = np.array(ts)
            pos_prop = k        
            
            p_w = pos_prop / len(pos)
            n_w = (1 - pos_prop) / len(neg)
            t_w = -1 / len(test)

            p = list(map(lambda x: (x, p_w), pos))
            n = list(map(lambda x: (x, n_w), neg))
            t = list(map(lambda x: (x, t_w), test))

            v = sorted(p + n + t, key = lambda x: x[0])

            acc = v[0][1] 
            total_cost = 0

            for i in range(1, len(v)):
                cost_mul = v[i][0] - v[i - 1][0] 
                total_cost = total_cost + abs(cost_mul * acc)
                acc = acc + v[i][1]

            vDist.append(total_cost)        
            
        prevalence = alpha[vDist.index(min(vDist))]
            
        return np.clip(prevalence, 0, 1)
        
    
    