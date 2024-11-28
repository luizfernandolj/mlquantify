import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ...base import AggregativeQuantifier

class EMQ(AggregativeQuantifier):
    """Expectation Maximisation Quantifier. It is a method that
    ajust the priors and posteriors probabilities of a learner
    """
    
    MAX_ITER = 1000
    EPSILON = 1e-6
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.priors = None
    
    def _fit_method(self, X, y):
        
        if not self.learner_fitted:
            self.learner.fit(X, y)
        
        counts = np.array([np.count_nonzero(y == _class) for _class in self.classes])
        self.priors = counts / len(y)
        
        return self
    
    def _predict_method(self, X) -> dict:
        
        posteriors = self.learner.predict_proba(X)
        prevalences, _ = self.EM(self.priors, posteriors)
        
        return prevalences
    
    
    def predict_proba(self, X, epsilon:float=EPSILON, max_iter:int=MAX_ITER) -> np.ndarray:
        posteriors = self.learner.predict_proba(X)
        _, posteriors = self.EM(self.priors, posteriors, epsilon, max_iter)
        return posteriors
    
    
    @classmethod
    def EM(cls, priors, posteriors, epsilon=EPSILON, max_iter=MAX_ITER):
        """Expectaion Maximization function, it iterates several times
        and At each iteration step, both the a posteriori and the a 
        priori probabilities are reestimated sequentially for each new 
        observation and each class. The iterative procedure proceeds 
        until the convergence of the estimated probabilities.

        Args:
            priors (array-like): priors probabilites of the train.
            posteriors (array-like): posteriors probabiblities of the test.
            epsilon (float): value that helps to indify the convergence.
            max_iter (int): max number of iterations.

        Returns:
            the predicted prevalence and the ajusted posteriors.
        """
        
        Px = posteriors
        prev_prevalence = np.copy(priors)
        running_estimate = np.copy(prev_prevalence)  # Initialized with the training prevalence

        iteration, converged = 0, False
        previous_estimate = None

        while not converged and iteration < max_iter:
            # E-step: ps is P(y|xi)
            posteriors_unnormalized = (running_estimate / prev_prevalence) * Px
            posteriors = posteriors_unnormalized / posteriors_unnormalized.sum(axis=1, keepdims=True)

            # M-step:
            running_estimate = posteriors.mean(axis=0)

            if previous_estimate is not None and np.mean(np.abs(running_estimate - previous_estimate)) < epsilon and iteration > 10:
                converged = True

            previous_estimate = running_estimate
            iteration += 1

        if not converged:
            print('[Warning] The method has reached the maximum number of iterations; it might not have converged')

        return running_estimate, posteriors
