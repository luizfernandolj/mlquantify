import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel
from ....utils import getHist

class HDy(MixtureModel):
    """Hellinger Distance Minimization. The method
    is based on computing the hellinger distance of 
    two distributions, test distribution and the mixture
    of the positive and negative distribution of the train.
    """

    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
    
        
    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        
        best_alphas, _ = self.GetMinDistancesHDy(test_scores)
        # Compute the median of the best alpha values as the final prevalence estimate
        prevalence = np.median(best_alphas)
            
        return prevalence
    
    
    
    def best_distance(self, X_test) -> float:
        
        test_scores = self.learner.predict_proba(X_test)
        
        _, distances = self.GetMinDistancesHDy(test_scores)
        
        size = len(distances)
        
        if size % 2 != 0:  # ODD
            index = size // 2
            distance = distances[index]
        else:  # EVEN
            # Find the two middle indices
            middle1 = np.floor(size / 2).astype(int)
            middle2 = np.ceil(size / 2).astype(int)

            # Get the values corresponding to the median positions
            dist1 = distances[middle1]
            dist2 = distances[middle2]
            
            # Calculate the average of the corresponding distances
            distance = np.mean([dist1, dist2])
        
        return distance
        

    def GetMinDistancesHDy(self, test_scores: np.ndarray) -> tuple:
        
        # Define bin sizes and alpha values
        bins_size = np.arange(10, 110, 11)  # Bins from 10 to 110 with a step size of 10
        alpha_values = np.round(np.linspace(0, 1, 101), 2)  # Alpha values from 0 to 1, rounded to 2 decimal places
        
        best_alphas = []
        distances = []
        
        for bins in bins_size:

            pos_bin_density = getHist(self.pos_scores, bins)
            neg_bin_density = getHist(self.neg_scores, bins)
            test_bin_density = getHist(test_scores, bins)
            
            distances = []
            
            # Evaluate distance for each alpha value
            for x in alpha_values:
                # Combine densities using a mixture of positive and negative densities
                train_combined_density = (pos_bin_density * x) + (neg_bin_density * (1 - x))
                # Compute the distance using the Hellinger measure
                distances.append(self.get_distance(train_combined_density, test_bin_density, measure="hellinger"))

            # Find the alpha value that minimizes the distance
            best_alphas.append(alpha_values[np.argmin(distances)])
            distances.append(min(distances)) 
            
        return best_alphas, distances