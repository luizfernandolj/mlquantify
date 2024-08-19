import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from .gac import GAC
from ...base import AggregativeQuantifier

class GPAC(AggregativeQuantifier):
    """Generalized Probabilistic Adjusted Count. Like 
    GAC, it also build a system of linear equations, but 
    utilize the confidence scores from probabilistic 
    classifiers as in the PAC method.
    """
    
    
    def __init__(self, learner: BaseEstimator, train_size:float=0.6, random_state:int=None):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.cond_prob_matrix = None
        self.train_size = train_size
        self.random_state = random_state
    
    def _fit_method(self, X, y):
        # Convert X and y to DataFrames if they are numpy arrays
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
            
        if self.learner_fitted:
            # Use existing model to predict
            y_pred = self.learner.predict(X)
            y_labels = y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_size, stratify=y, random_state=self.random_state
            )
            
            self.learner.fit(X_train, y_train)
            
            y_labels = y_val
            y_pred = self.learner.predict(X_val)
        
        # Compute conditional probability matrix using GAC
        self.cond_prob_matrix = GAC.get_cond_prob_matrix(self.classes, y_labels, y_pred)
        
        return self
    
    def _predict_method(self, X) -> dict:
        # Predict class labels for the test data
        predictions = self.learner.predict(X)

        # Calculate the distribution of predictions in the test set
        predicted_prevalences = np.zeros(self.n_class)
        _, counts = np.unique(predictions, return_counts=True)
        predicted_prevalences[:len(counts)] = counts
        predicted_prevalences = predicted_prevalences / predicted_prevalences.sum()
        
        # Adjust prevalences based on the conditional probability matrix from GAC
        adjusted_prevalences = GAC.solve_adjustment(self.cond_prob_matrix, predicted_prevalences)

        # Map class labels to their corresponding prevalences
        return adjusted_prevalences
    
    @classmethod
    def get_cond_prob_matrix(cls, classes:list, y_labels:np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        """Estimate the matrix where entry (i,j) is the estimate of P(yi|yj)"""
        
        n_classes = len(classes)
        cond_prob_matrix = np.eye(n_classes)
        
        for i, class_ in enumerate(classes):
            class_indices = y_labels == class_
            if class_indices.any():
                cond_prob_matrix[i] = y_pred[class_indices].mean(axis=0)
        
        return cond_prob_matrix.T
