import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel
from ....utils import getHist, ternary_search

class DyS(MixtureModel):
    """Distribution y-Similarity framework. Is a 
    method that generalises the HDy approach by 
    considering the dissimilarity function DS as 
    a parameter of the model
    """
    
    def __init__(self, learner:BaseEstimator, measure:str="topsoe", bins_size:np.ndarray=None):
        assert measure in ["hellinger", "topsoe", "probsymm"], "measure not valid"
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
        # Set up bins_size
        if not bins_size:
            bins_size = np.append(np.linspace(2,20,10), 30)
        if isinstance(bins_size, list):
            bins_size = np.asarray(bins_size)
            
        self.bins_size = bins_size
        self.measure = measure
        self.prevs = None # Array of prevalences that minimizes the distances
        
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:    
        
        prevs = self.GetMinDistancesDyS(test_scores)                    
        # Use the median of the prevalences as the final prevalence estimate
        prevalence = np.median(prevs)
            
        return prevalence
    
    
    
    def best_distance(self, X_test) -> float:
        
        test_scores = self.learner.predict_proba(X_test)
        
        prevs = self.GetMinDistancesDyS(test_scores) 
        
        size = len(prevs)
        best_prev = np.median(prevs)

        if size % 2 != 0:  # ODD
            index = np.argmax(prevs == best_prev)
            bin_size = self.bins_size[index]
        else:  # EVEN
            # Sort the values in self.prevs
            ordered_prevs = np.sort(prevs)

            # Find the two middle indices
            middle1 = np.floor(size / 2).astype(int)
            middle2 = np.ceil(size / 2).astype(int)

            # Get the values corresponding to the median positions
            median1 = ordered_prevs[middle1]
            median2 = ordered_prevs[middle2]

            # Find the indices of median1 and median2 in prevs
            index1 = np.argmax(prevs == median1)
            index2 = np.argmax(prevs == median2)

            # Calculate the average of the corresponding bin sizes
            bin_size = np.mean([self.bins_size[index1], self.bins_size[index2]])
            
        
        pos_bin_density = getHist(self.pos_scores, bin_size)
        neg_bin_density = getHist(self.neg_scores, bin_size)
        test_bin_density = getHist(test_scores, bin_size)
        
        train_combined_density = (pos_bin_density * best_prev) + (neg_bin_density * (1 - best_prev))
        
        distance = self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
        
        return distance
        

    def GetMinDistancesDyS(self, test_scores) -> list:
        # Compute prevalence by evaluating the distance metric across various bin sizes
        
        prevs = []
 
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
            prevs.append(ternary_search(0, 1, f))
            
        return prevs
        
        