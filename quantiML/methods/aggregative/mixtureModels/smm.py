import numpy as np
from sklearn.base import BaseEstimator

from ._MixtureModel import MixtureModel

class SMM(MixtureModel):
    """
    Implementation of Hellinger Distance-based Quantifier (HDy)
    """
    
    def __init__(self, learner:BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        super().__init__(learner)
        
    
    def _compute_prevalence(self, test_scores:np.ndarray) -> float:
        mean_pos_score = np.mean(self.pos_scores)
        mean_neg_score = np.mean(self.neg_scores)  #calculating mean of pos & neg scores
    
        mean_test_score = np.mean(test_scores)              #Mean of test scores
            
        prevalence =  (mean_test_score - mean_neg_score)/(mean_pos_score - mean_neg_score)   #evaluating Positive class proportion
        
        return prevalence