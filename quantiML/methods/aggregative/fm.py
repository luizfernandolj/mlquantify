import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

from ...base import AggregativeQuantifier
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
        counts = np.array([np.count_nonzero(y_labels == _class) for _class in self.classes])
        self.prior = counts / len(y_labels)
        
        for i, _class in enumerate(self.classes):       
            idx = np.where(y_labels == _class)[0]
            CM[:, i] = np.sum(probabilities[idx] > self.prior, axis=0) 
        self.CM = CM / counts
        
        return self
    
    
    
    def _predict_method(self, X) -> dict:
        prevalences = {}
        
        test_scores = self.learner.predict_proba(X)
        p_y_hat = np.sum(test_scores > self.prior, axis=0) / test_scores.shape[0]
        
        def objective(p_hat):
            return np.linalg.norm(self.CM @ p_hat - p_y_hat)
        
        # Constraints for the optimization problem
        cons = ({'type': 'eq', 'fun': lambda p_hat: np.sum(p_hat) - 1.0},
                {'type': 'ineq', 'fun': lambda p_hat: p_hat})
        
        # Initial guess
        p_hat_initial = np.ones(self.CM.shape[1]) / self.CM.shape[1]
        
        # Solve the optimization problem
        result = minimize(objective, p_hat_initial, constraints=cons, bounds=[(0, 1)]*self.CM.shape[1])
        
        if result.success:
            p_hat = result.x
        else:
            raise ValueError("Optimization did not converge")
        
        prevalences = {_class: prevalence for _class, prevalence in zip(self.classes, p_hat)}
        
        return prevalences