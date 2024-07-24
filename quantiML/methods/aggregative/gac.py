import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from ...base import AggregativeQuantifier


class GAC(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.cond_prob_matrix = None
    
    def _fit_method(self, X, y, learner_fitted: bool = False, cv_folds: int = 10):
        # Ensure X and y are DataFrames
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
            
        if learner_fitted:
            y_pred = self.learner.predict(X)
            y_label = y
        else:
            # Cross-validation for generating predictions
            skf = StratifiedKFold(n_splits=cv_folds)
            y_pred = []
            y_label = []
            
            for train_index, valid_index in skf.split(X, y):
                train_data = pd.DataFrame(X.iloc[train_index])
                train_label = y.iloc[train_index]
                
                valid_data = pd.DataFrame(X.iloc[valid_index])
                valid_label = y.iloc[valid_index]
                
                self.learner.fit(train_data, train_label)
                
                y_pred.extend(self.learner.predict(valid_data))
                y_label.extend(valid_label)
        
        # Compute conditional probability matrix
        self.cond_prob_matrix = self.get_cond_prob_matrix(self.classes, y, y_pred)
        
        return self
    
    def _predict_method(self, X) -> dict:
        # Predict class labels for the test data
        y_pred = self.learner.predict(X)

        # Distribution of predictions in the test set
        _, counts = np.unique(y_pred, return_counts=True)
        predicted_prevalences = counts / counts.sum()
        
        # Adjust prevalences based on conditional probability matrix
        adjusted_prevalences = self.solve_adjustment(self.cond_prob_matrix, predicted_prevalences)

        return {_class: prevalence for _class, prevalence in zip(self.classes, adjusted_prevalences)}
    
    @classmethod
    def get_cond_prob_matrix(cls, classes, y, y_pred):
        # Estimate the conditional probability matrix P(yi|yj)
        CM = confusion_matrix(y, y_pred, labels=classes).T
        CM = CM.astype(np.float32)
        class_counts = CM.sum(axis=0)
        for i, _ in enumerate(classes):
            if class_counts[i] == 0:
                CM[i, i] = 1
            else:
                CM[:, i] /= class_counts[i]
        return CM
    
    @classmethod
    def solve_adjustment(cls, cond_prob_matrix, predicted_prevalences):
        # Solve the linear system Ax = B with A=cond_prob_matrix and B=predicted_prevalences
        A = cond_prob_matrix
        B = predicted_prevalences
        try:
            adjusted_prevalences = np.linalg.solve(A, B)
            adjusted_prevalences = np.clip(adjusted_prevalences, 0, 1)
            adjusted_prevalences /= adjusted_prevalences.sum()
        except (np.linalg.LinAlgError, ValueError):
            adjusted_prevalences = predicted_prevalences  # No way to adjust them
        return adjusted_prevalences
