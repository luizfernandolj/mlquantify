import numpy as np
from sklearn.base import BaseEstimator

from ...base import AggregativeQuantifier

class PCC(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
        
    def _fit_method(self, X, y, learner_fitted:bool=False, cv_folds:int=10):
        
        self.learner.fit(X, y)
        
        return self
    
    def _predict_method(self, X) -> dict:
        prevalences = {}
        
        for i, _class in enumerate(self.classes):
            test_scores = self.learner.predict_proba(X)[:, i]
        
            prevalence = np.mean(test_scores)
            prevalences[_class] = prevalence
        
        return prevalences