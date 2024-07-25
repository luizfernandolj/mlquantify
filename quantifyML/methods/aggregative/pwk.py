import numpy as np
from sklearn.base import BaseEstimator
from collections import defaultdict

from ...base import AggregativeQuantifier

class PWK(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
    def _fit_method(self, X, y):
        if not self.learner_fitted:
            self.learner.fit(X, y)
        return self
    
    def _predict_method(self, X) -> dict:
        prevalences = {_class:0 for _class in self.classes}
        
        # Predict class labels for the given data
        predicted_labels = self.learner.predict(X)
        
        # Compute the distribution of predicted labels
        unique_labels, label_counts = np.unique(predicted_labels, return_counts=True)
        
        # Calculate the prevalence for each class
        class_prevalences = label_counts / label_counts.sum()
        
        # Map each class label to its prevalence
        prevalences  = {label: prevalence for label, prevalence in zip(unique_labels, class_prevalences)}
        
        prevalences = defaultdict(lambda: 0, prevalences)
    
        # Ensure all classes are present in the result
        for cls in self.classes:
            prevalences[cls] = prevalences[cls]  # This will ensure the class is present and initialize with 0 if not already in the dictionary
        
        return dict(prevalences)
