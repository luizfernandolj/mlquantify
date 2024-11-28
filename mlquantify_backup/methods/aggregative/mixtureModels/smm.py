import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel

class SMM(MixtureModel):
    """Sample Mean Matching. The method is 
    a member of the DyS framework that uses 
    simple means to represent the score 
    distribution for positive, negative 
    and unlabelled scores.
    """

    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
    def _compute_prevalence(self, test_scores: np.ndarray) -> float:
        mean_pos_score = np.mean(self.pos_scores)
        mean_neg_score = np.mean(self.neg_scores)
        mean_test_score = np.mean(test_scores)
        
        # Calculate prevalence as the proportion of the positive class
        # based on the mean test score relative to the mean positive and negative scores
        prevalence = (mean_test_score - mean_neg_score) / (mean_pos_score - mean_neg_score)
        
        return prevalence
