import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from ...base import AggregativeQuantifier
from sklearn.model_selection import StratifiedKFold
from ...utils import GetScores

class FM(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.CM = None
    
    def _fit_method(self, X, y, learner_fitted: bool = False, cv_folds: int = 10):
        y_label, probabilities = GetScores(X, y, self.learner, cv_folds, learner_fitted)
        self.learner.fit(X, y) if learner_fitted is False else None
        
        CM = np.zeros((self.n_class, self.n_class))

        # Count the occurrences of each class in the training labels
        class_counts = np.array([np.count_nonzero(y_label == _class) for _class in self.classes])

        # Calculate the prior probabilities of each class
        priors = class_counts / len(y_label)

        # Populate the confusion matrix
        for i, _class in enumerate(self.classes):
            idx = np.where(y_label == _class)[0]
            CM[:, i] += np.sum(probabilities[idx] > priors, axis=0)
        CM = CM / class_counts

        self.CM = CM
        
        return self
    
    def _predict_method(self, X) -> dict:
        # Initialize the confusion matrix

        # Estimate the distribution of predictions in the test set
        p_y_hat = np.sum(test_scores > p_yt, axis=0) / test_scores.shape[0]

        # Solve the linear system to adjust the prevalences
        try:
            p_hat = np.linalg.solve(CM, p_y_hat)
            p_hat = np.clip(p_hat, 0, 1)
            p_hat /= p_hat.sum()
        except np.linalg.LinAlgError:
            p_hat = p_y_hat  # If the system cannot be solved, use the initial estimated prevalences
        except ValueError:
            p_hat = p_y_hat  # If there is a value error (e.g., due to mismatched dimensions), use the initial estimated prevalences
            print('Error: Unable to adjust prevalences due to dimension mismatch or singular matrix.')

        # Return prevalences in dictionary format
        prevalences = {i: p_hat[i] for i in range(nclasses)}
        return prevalences