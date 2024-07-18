import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel
from ....utils import getHist, ternary_search

class HDy(MixtureModel):
    """
    Implementation of Hellinger Distance-based Quantifier (HDy)
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:
        bin_size = np.linspace(10,110,11)       #creating bins from 10 to 110 with step size 10
    #alpha_values = [round(x, 2) for x in np.linspace(0,1,101)]
        alpha_values = np.round(np.linspace(0,1,101), 2)
        
        best_alphas = []
 
        for bins in bin_size:
            
            pos_bin_density = getHist(self.pos_scores, bins)
            neg_bin_density = getHist(self.neg_scores, bins)
            test_bin_density = getHist(test_scores, bins) 

            distances = []
            
            for x in alpha_values:
                train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                distances.append(self.get_distance(train_combined_density, test_bin_density, measure="hellinger"))

            best_alphas.append(alpha_values[np.argmin(distances)])
        
        prevalence = np.median(best_alphas)
            
        return prevalence
        