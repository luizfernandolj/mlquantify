import numpy as np
from sklearn.base import BaseEstimator
from ...base import AggregativeQuantifier

class CC(AggregativeQuantifier):
    """Classify and Count. The simplest quantification method
    involves classifying each instance and then counting the 
    number of instances assigned to each class to estimate 
    the class prevalence.
    """
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
    
    def _fit_method(self, X, y):
        if not self.learner_fitted:
            self.learner.fit(X, y)
        return self
    
    
    def _predict_method(self, X) -> dict:
        predicted_labels = self.learner.predict(X)
        
        # Count occurrences of each class in the predictions
        class_counts = np.array([np.count_nonzero(predicted_labels == _class) for _class in self.classes])
        
        # Calculate the prevalence of each class
        prevalences = class_counts / len(predicted_labels)
        
        return prevalences
