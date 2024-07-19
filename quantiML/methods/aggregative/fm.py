import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
import cvxpy as cvx

from ...base import AggregativeQuantifier
from sklearn.model_selection import StratifiedKFold
from ...utils import GetScores

class FM(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.CM = None
    
    def _fit_method(self, X, y, learner_fitted: bool = False, cv_folds: int = 10):
        y_labels, probabilities = GetScores(X, y, self.learner, cv_folds, learner_fitted)
        self.learner.fit(X, y) if learner_fitted is False else None
        
        CM = np.zeros((self.n_class, self.n_class))
        y_cts = np.array([np.count_nonzero(y_labels == _class) for _class in self.classes])
        self.p_yt = y_cts / len(y_labels)
        
        for i, _class in enumerate(self.classes):       
            idx = np.where(y_labels == _class)[0]
            CM[:, i] = np.sum(probabilities[idx] > self.p_yt, axis=0) 
        self.CM = CM / y_cts
        
        return self
    
    def _predict_method(self, X) -> dict:
        prevalences = {}
        
        test_scores = self.learner.predict_proba(X)
        p_y_hat = np.sum(test_scores > self.p_yt, axis = 0) / test_scores.shape[0]
        
        p_hat = cvx.Variable(self.CM.shape[1])
        constraints = [p_hat >= 0, cvx.sum(p_hat) == 1.0]
        problem = cvx.Problem(cvx.Minimize(cvx.norm(self.CM @ p_hat - p_y_hat)), constraints)
        problem.solve()
        
        prevalences = {_class:prevalence for _class, prevalence in zip(self.classes, p_hat.value)}
        
        
        return prevalences