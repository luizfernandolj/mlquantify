import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel
from ....utils import getHist, ternary_search, MoSS, get_real_prev

class DySsyn(MixtureModel):
    """Synthetic Distribution y-Similarity. This method works the
    same as DyS method, but istead of using the train scores, it 
    generates them via MoSS (Model for Score Simulation) which 
    generate a spectrum of score distributions from highly separated
    scores to fully mixed scores.
    """
    
    def __init__(self, learner:BaseEstimator, measure:str="topsoe", merge_factor:np.ndarray=None, bins_size:np.ndarray=None, alpha_train:float=0.5, n:int=None):
        assert measure in ["hellinger", "topsoe", "probsymm"], "measure not valid"
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
        # Set up bins_size
        if not bins_size:
            bins_size = np.append(np.linspace(2,20,10), 30)
        if isinstance(bins_size, list):
            bins_size = np.asarray(bins_size)
            
        if not merge_factor:
            merge_factor = np.linspace(0.1, 0.4, 10)
            
        self.bins_size = bins_size
        self.merge_factor = merge_factor
        self.alpha_train = alpha_train
        self.n = n
        self.measure = measure
        self.m = None
    
    
    
    def _fit_method(self, X, y):
        if not self.learner_fitted:
            self.learner.fit(X, y)
            
        self.alpha_train = list(get_real_prev(y).values())[1]
        
        return self
    
    
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:    #creating bins from 10 to 110 with step size 10
        
        distances = self.GetMinDistancesDySsyn(test_scores)
        
        # Use the median of the prevss as the final prevalence estimate
        index = min(distances, key=lambda d: distances[d][0])
        prevalence = distances[index][1]
            
        return prevalence
    
    
    def best_distance(self, X_test):
        
        test_scores = self.learner.predict_proba(X_test)
        
        distances = self.GetMinDistancesDySsyn(test_scores)
        
        index = min(distances, key=lambda d: distances[d][0])
        
        distance = distances[index][0]
        
        return distance
    
    

    def GetMinDistancesDySsyn(self, test_scores) -> list:
        # Compute prevalence by evaluating the distance metric across various bin sizes
        if self.n is None:
            self.n = len(test_scores)
            
        values = {}
        
        # Iterate over each bin size
        for m in self.merge_factor:
            pos_scores, neg_scores = MoSS(self.n, self.alpha_train, m)
            prevs  = []
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
                prevs.append(ternary_search(0, 1, f))
                
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
                
            
            pos_bin_density = getHist(pos_scores, bin_size)
            neg_bin_density = getHist(neg_scores, bin_size)
            test_bin_density = getHist(test_scores, bin_size)
            
            train_combined_density = (pos_bin_density * best_prev) + (neg_bin_density * (1 - best_prev))
            
            distance = self.get_distance(train_combined_density, test_bin_density, measure=self.measure)
            
            values[m] = (distance, best_prev)
            
        return values