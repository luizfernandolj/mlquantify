import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel
from ....utils import getHist, ternary_search

class SORD(MixtureModel):
    """
    Implementation of Hellinger Distance-based Quantifier (HDy)
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
    
    def _compute_prevalence(self, pos_scores:np.ndarray, neg_scores:np.ndarray, test_scores:np.ndarray, measure:str) -> float:
        alpha = np.linspace(0,1,101)
        
        vDist   = []
        for k in alpha:        
            pos = np.array(pos_scores)
            neg = np.array(neg_scores)
            test = np.array(test_scores)
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