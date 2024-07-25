import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

from .gac import GAC
from ...base import AggregativeQuantifier

class GPAC(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.cond_prob_matrix = None
    
    def _fit_method(self, X, y):
        # Convert X and y to DataFrames if they are numpy arrays
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
            
        if self.learner_fitted:
            # Use existing model to predict
            predictions = self.learner.predict(X)
            true_labels = y
        else:
            # Perform cross-validation to generate predictions
            skf = StratifiedKFold(n_splits=self.cv_folds)
            predictions = []
            true_labels = []
            
            for train_index, valid_index in skf.split(X, y):
                # Split data into training and validation sets
                train_data = pd.DataFrame(X.iloc[train_index])
                train_labels = y.iloc[train_index]
                
                valid_data = pd.DataFrame(X.iloc[valid_index])
                valid_labels = y.iloc[valid_index]
                
                # Train the learner
                self.learner.fit(train_data, train_labels)
                
                # Predict and collect results
                predictions.extend(self.learner.predict(valid_data))
                true_labels.extend(valid_labels)
        
        # Compute conditional probability matrix using GAC
        self.cond_prob_matrix = GAC.get_cond_prob_matrix(self.classes, true_labels, predictions)
        
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
    def get_cond_prob_matrix(cls, classes, true_labels, predictions):
        # Estimate the matrix where entry (i,j) is the estimate of P(yi|yj)
        n_classes = len(classes)
        cond_prob_matrix = np.eye(n_classes)
        
        for i, class_ in enumerate(classes):
            class_indices = true_labels == class_
            if class_indices.any():
                cond_prob_matrix[i] = predictions[class_indices].mean(axis=0)
        
        return cond_prob_matrix.T
