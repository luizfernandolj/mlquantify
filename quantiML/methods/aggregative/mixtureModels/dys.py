import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel
from ....utils import getHist, ternary_search

class DyS(MixtureModel):
    """
    Implementation of Hellinger Distance-based Quantifier (HDy)
    """
    
    def __init__(self, learner:BaseEstimator, measure:str="topsoe", bins_size:np.ndarray=None):
        assert measure in ["hellinger", "topsoe", "probsymm"], "measure not valid"
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
        # Set up bins_size
        if not bins_size:
            bins_size = np.append(np.linspace(2,20,10), 30)
        if isinstance(bins_size, int):
            bins_size = np.linspace(2,bins_size,10)
        if len(bins_size) == 2:
            bins_size = np.linspace(bins_size[0],bins_size[1],10)
        if len(bins_size) == 3:
            bins_size = np.linspace(bins_size[0],bins_size[1],bins_size[2])
            
        self.bins_size = bins_size
        self.measure = measure
        
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:    #creating bins from 10 to 110 with step size 10
        # Compute prevalence by evaluating the distance metric across various bin sizes
        
        result = []
 
        # Iterate over each bin size
        for bins in self.bins_size:
            # Compute histogram densities for positive, negative, and test scores
            pos_bin_density = getHist(self.pos_scores, bins)
            neg_bin_density = getHist(self.neg_scores, bins)
            test_bin_density = getHist(test_scores, bins)

            # Define the function to minimize
            def f(x):
                # Combine densities using a mixture of positive and negative densities
                train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                # Calculate the distance between combined density and test density
                return self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
        
            # Use ternary search to find the best x that minimizes the distance
            result.append(ternary_search(0, 1, f))
                            
        # Use the median of the results as the final prevalence estimate
        prevalence = np.median(result)
            
        return prevalence
        