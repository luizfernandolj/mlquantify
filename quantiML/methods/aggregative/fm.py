import numpy as np
from sklearn.base import BaseEstimator
from scipy.optimize import minimize

from ...base import AggregativeQuantifier
from ...utils import get_scores

class FM(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.CM = None
    
    def _fit_method(self, X, y, learner_fitted: bool = False, cv_folds: int = 10):
        # Get predicted labels and probabilities using cross-validation
        y_labels, probabilities = get_scores(X, y, self.learner, cv_folds, learner_fitted)
        
        # Fit the learner if it hasn't been fitted already
        if not learner_fitted:
            self.learner.fit(X, y)
        
        # Initialize the confusion matrix
        CM = np.zeros((self.n_class, self.n_class))
        
        # Calculate the class priors
        class_counts = np.array([np.count_nonzero(y_labels == _class) for _class in self.classes])
        self.priors = class_counts / len(y_labels)
        
        # Populate the confusion matrix
        for i, _class in enumerate(self.classes):       
            indices = np.where(y_labels == _class)[0]
            CM[:, i] = np.sum(probabilities[indices] > self.priors, axis=0) 
        
        # Normalize the confusion matrix by class counts
        self.CM = CM / class_counts
        
        return self
    
    def _predict_method(self, X) -> dict:
        posteriors = self.learner.predict_proba(X)
        
        # Calculate the estimated prevalences in the test set
        prevs_estim = np.sum(posteriors > self.priors, axis=0) / posteriors.shape[0]
        # Define the objective function for optimization
        def objective(prevs_pred):
            return np.linalg.norm(self.CM @ prevs_pred - prevs_estim)
        
        # Constraints for the optimization problem
        constraints = [{'type': 'eq', 'fun': lambda prevs_pred: np.sum(prevs_pred) - 1.0},
                       {'type': 'ineq', 'fun': lambda prevs_pred: prevs_pred}]
        
        # Initial guess for the optimization
        initial_guess = np.ones(self.CM.shape[1]) / self.CM.shape[1]
        
        # Solve the optimization problem
        result = minimize(objective, initial_guess, constraints=constraints, bounds=[(0, 1)]*self.CM.shape[1])
        
        if result.success:
            prevalences = result.x
        else:
            print("Optimization did not converge")
            prevalences = self.priors

        prevalences = {_class: prevalence for _class, prevalence in zip(self.classes, prevalences)}
        
        return prevalences