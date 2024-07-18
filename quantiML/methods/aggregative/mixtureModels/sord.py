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
        self.distance = None
        
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:
        alpha, distances = self._calculate_distances(test_scores)
        
        self.distance = np.argmin(distances)
        prevalence = alpha[self.distance]
        
        return prevalence
    
    def _calculate_distances(self, test_scores):
        alpha = np.linspace(0, 1, 101)
        pos_len = len(self.pos_scores)
        neg_len = len(self.neg_scores)
        test_len = len(test_scores)

        distances = []

        for k in alpha:
            pos_prop = k
            p_w = pos_prop / pos_len
            n_w = (1 - pos_prop) / neg_len
            t_w = -1 / test_len

            pos_weights = np.full(pos_len, p_w)
            neg_weights = np.full(neg_len, n_w)
            test_weights = np.full(test_len, t_w)

            scores = np.concatenate([self.pos_scores, self.neg_scores, test_scores])
            weights = np.concatenate([pos_weights, neg_weights, test_weights])

            sorted_indices = np.argsort(scores)
            sorted_scores = scores[sorted_indices]
            sorted_weights = weights[sorted_indices]

            acc = sorted_weights[0]
            total_cost = 0

            for i in range(1, len(sorted_scores)):
                cost_mul = sorted_scores[i] - sorted_scores[i - 1]
                total_cost += abs(cost_mul * acc)
                acc += sorted_weights[i]

            distances.append(total_cost)

        return alpha, distances