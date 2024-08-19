import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from ...base import AggregativeQuantifier


class GAC(AggregativeQuantifier):
    """Generalized Adjusted Count. It applies a 
    classifier to build a system of linear equations, 
    and solve it via constrained least-squares regression.
    """
    
    def __init__(self, learner: BaseEstimator, train_size:float=0.6, random_state:int=None):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.cond_prob_matrix = None
        self.train_size = train_size
        self.random_state = random_state
    
    def _fit_method(self, X, y):
        # Ensure X and y are DataFrames
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)

        if self.learner_fitted:
            y_pred = self.learner.predict(X)
            y_label = y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, train_size=self.train_size, stratify=y, random_state=self.random_state
            )
            
            self.learner.fit(X_train, y_train)
            
            y_label = y_val
            y_pred = self.learner.predict(X_val)
        
        # Compute conditional probability matrix
        self.cond_prob_matrix = GAC.get_cond_prob_matrix(self.classes, y_label, y_pred)
        
        return self
    
    def _predict_method(self, X) -> dict:
        # Predict class labels for the test data
        y_pred = self.learner.predict(X)

        # Distribution of predictions in the test set
        _, counts = np.unique(y_pred, return_counts=True)
        predicted_prevalences = counts / counts.sum()
        
        # Adjust prevalences based on conditional probability matrix
        adjusted_prevalences = self.solve_adjustment(self.cond_prob_matrix, predicted_prevalences)

        return adjusted_prevalences
    
    @classmethod
    def get_cond_prob_matrix(cls, classes:list, y_labels:np.ndarray, predictions:np.ndarray) -> np.ndarray:
        """ Estimate the conditional probability matrix P(yi|yj)"""

        CM = confusion_matrix(y_labels, predictions, labels=classes).T
        CM = CM.astype(float)
        class_counts = CM.sum(axis=0)
        for i, _ in enumerate(classes):
            if class_counts[i] == 0:
                CM[i, i] = 1
            else:
                CM[:, i] /= class_counts[i]
        return CM
    
    @classmethod
    def solve_adjustment(cls, cond_prob_matrix, predicted_prevalences):
        """ Solve the linear system Ax = B with A=cond_prob_matrix and B=predicted_prevalences
        """
        
        #
        A = cond_prob_matrix
        B = predicted_prevalences
        try:
            adjusted_prevalences = np.linalg.solve(A, B)
            adjusted_prevalences = np.clip(adjusted_prevalences, 0, 1)
            adjusted_prevalences /= adjusted_prevalences.sum()
        except (np.linalg.LinAlgError):
            adjusted_prevalences = predicted_prevalences  # No way to adjust them
        return adjusted_prevalences
