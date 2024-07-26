import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel
from ....utils import getHist, ternary_search, MoSS

class DySsyn(MixtureModel):
    """
    Implementation of Hellinger Distance-based Quantifier (HDy)
    """
    
    def __init__(self, learner:BaseEstimator, measure:str="topsoe", bins_size:np.ndarray=None, alpha_train:float=0.5, n:int=None):
        assert measure in ["hellinger", "topsoe", "probsymm"], "measure not valid"
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
        # Set up bins_size
        if not bins_size:
            bins_size = np.append(np.linspace(2,20,10), 30)
        if isinstance(bins_size, list):
            bins_size = np.asarray(bins_size)
            
        self.bins_size = bins_size
        self.alpha_train = alpha_train
        self.n = n
        self.measure = measure
        self.m = None
        
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:    #creating bins from 10 to 110 with step size 10
        # Compute prevalence by evaluating the distance metric across various bin sizes
        if self.n is None:
            self.n = len(test_scores)
            
        distances = {}
        
        # Iterate over each bin size
        for m in np.linspace(0.1, 0.4, 10):
            pos_scores, neg_scores = MoSS(self.n, self.alpha_train, m)
            result  = []
            for bins in self.bins_size:
                # Compute histogram densities for positive, negative, and test scores
                pos_bin_density = getHist(pos_scores, bins)
                neg_bin_density = getHist(neg_scores, bins)
                test_bin_density = getHist(test_scores, bins)

                # Define the function to minimize
                def f(x):
                    # Combine densities using a mixture of positive and negative densities
                    train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                    # Calculate the distance between combined density and test density
                    return self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
            
                # Use ternary search to find the best x that minimizes the distance
                result.append(ternary_search(0, 1, f))
            prevalence = np.median(result)
            
            bins_size = self.bins_size[result == prevalence][0]
            
            pos_bin_density = getHist(pos_scores, bins_size)
            neg_bin_density = getHist(neg_scores, bins_size)
            test_bin_density = getHist(test_scores, bins_size)
            
            train_combined_density = (pos_bin_density * prevalence) + (neg_bin_density * (1 - prevalence))
            d = self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
            distances[m] = (d, prevalence)
        # Use the median of the results as the final prevalence estimate
        index = min(distances, key=lambda d: distances[d][0])
        prevalence = distances[index][1]
            
        return prevalence
        