import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel
from ....utils import getHist, ternary_search

class DyS(MixtureModel):
    """
    Implementation of Hellinger Distance-based Quantifier (HDy)
    """
    
    def __init__(self, learner:BaseEstimator, measure:str="topsoe"):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner, measure)
        
    
    def _compute_prevalence(self, pos_scores:np.ndarray, neg_scores:np.ndarray, test_scores:np.ndarray, measure:str) -> float:
        bin_size = np.linspace(10,110,11)       #creating bins from 10 to 110 with step size 10
    #alpha_values = [round(x, 2) for x in np.linspace(0,1,101)]
        alpha_values = np.linspace(0,1,101)
        
        result = []
 
        for bins in bin_size:
            
            pos_bin_density = getHist(pos_scores, bins)
            neg_bin_density = getHist(neg_scores, bins)
            test_bin_density = getHist(test_scores, bins) 

            def f(x):            
                return(self.get_distance(((pos_bin_density*x) + (neg_bin_density*(1-x))), test_bin_density, measure=self.measure))
        
            result.append(ternary_search(0, 1, f))                                           
                            
        prevalence = np.median(result)
            
        return np.clip(prevalence, 0, 1)
        