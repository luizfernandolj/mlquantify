import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel

class SORD(MixtureModel):
    """Sample Ordinal Distance. Is a method 
    that does not rely on distributions, but 
    estimates the prevalence of the positive 
    class in a test dataset by calculating and 
    minimizing a sample ordinal distance measure 
    between the test scores and known positive 
    and negative scores.
    """

    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
        self.best_distance_index = None
        
    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        # Compute alpha values and corresponding distance measures
        alpha_values, distance_measures = self._calculate_distances(test_scores)
        
        # Find the index of the alpha value with the minimum distance measure
        self.best_distance_index = np.argmin(distance_measures)
        prevalence = alpha_values[self.best_distance_index]
        
        return prevalence
    
    
    def _calculate_distances(self, test_scores: np.ndarray):
        # Define a range of alpha values from 0 to 1
        alpha_values = np.linspace(0, 1, 101)
        
        # Get the number of positive, negative, and test scores
        num_pos_scores = len(self.pos_scores)
        num_neg_scores = len(self.neg_scores)
        num_test_scores = len(test_scores)

        distance_measures = []

        # Iterate over each alpha value
        for alpha in alpha_values:
            # Compute weights for positive, negative, and test scores
            pos_weight = alpha / num_pos_scores
            neg_weight = (1 - alpha) / num_neg_scores
            test_weight = -1 / num_test_scores

            # Create arrays with weights
            pos_weights = np.full(num_pos_scores, pos_weight)
            neg_weights = np.full(num_neg_scores, neg_weight)
            test_weights = np.full(num_test_scores, test_weight)

            # Concatenate all scores and their corresponding weights
            all_scores = np.concatenate([self.pos_scores, self.neg_scores, test_scores])
            all_weights = np.concatenate([pos_weights, neg_weights, test_weights])

            # Sort scores and weights based on scores
            sorted_indices = np.argsort(all_scores)
            sorted_scores = all_scores[sorted_indices]
            sorted_weights = all_weights[sorted_indices]

            # Compute the total cost for the current alpha
            cumulative_weight = sorted_weights[0]
            total_cost = 0

            for i in range(1, len(sorted_scores)):
                # Calculate the cost for the segment between sorted scores
                segment_width = sorted_scores[i] - sorted_scores[i - 1]
                total_cost += abs(segment_width * cumulative_weight)
                cumulative_weight += sorted_weights[i]

            distance_measures.append(total_cost)

        return alpha_values, distance_measures
