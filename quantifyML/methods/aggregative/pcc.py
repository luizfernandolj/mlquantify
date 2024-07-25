import numpy as np
from sklearn.base import BaseEstimator
from ...base import AggregativeQuantifier

class PCC(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
    def _fit_method(self, X, y):
        if not self.learner_fitted:
            self.learner.fit(X, y)
        return self
    
    def _predict_method(self, X) -> dict:
        # Initialize a dictionary to store the prevalence for each class
        prevalences = []
        
        # Calculate the prevalence for each class
        for class_index, class_label in enumerate(self.classes):
            # Get the predicted probabilities for the current class
            class_probabilities = self.learner.predict_proba(X)[:, class_index]
        
            # Compute the average probability (prevalence) for the current class
            mean_prev = np.mean(class_probabilities)
            prevalences.append(mean_prev)
        
        return prevalences
