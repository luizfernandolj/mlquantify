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
        # Define bin sizes and alpha values
        bin_size = np.arange(10, 110, 11)  # Bins from 10 to 110 with a step size of 10
        alpha_values = np.round(np.linspace(0, 1, 101), 2)  # Alpha values from 0 to 1, rounded to 2 decimal places
        
        best_alphas = []

        for bins in bin_size:

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
        
        # Compute the median of the best alpha values as the final prevalence estimate
        prevalence = np.median(best_alphas)
            
        return prevalence
