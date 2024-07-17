import numpy as np
from sklearn.base import BaseEstimator

from ...base import AggregativeQuantifier

class CC(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
        
    def _fit_method(self, X, y, learner_fitted:bool=False, cv_folds:int=10):
        
        self.learner.fit(X, y)
        
        return self
    
    def _predict_method(self, X) -> dict:
        y_pred = self.learner.predict(X)
        classes, nclasses = np.unique(y_pred, return_counts=True)
        
        return { _class : nclass/len(y_pred) for _class, nclass in zip(classes, nclasses) }