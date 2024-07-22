import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from ...base import AggregativeQuantifier
from sklearn.model_selection import StratifiedKFold

class EMQ(AggregativeQuantifier):
    
    MAX_ITER = 1000
    EPSILON = 1e-6
    
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
    
    def _fit_method(self, X, y, learner_fitted: bool = False, cv_folds: int = 10):
        
        
        return self
    
    def _predict_method(self, X) -> dict:
        prevalences = {}
        
        y_pred = self.learner.predict(X)

        # Distribution of predictions in the test set
        prevs_estim = np.zeros(self.n_class)
        _, counts = np.unique(y_pred, return_counts=True)
        prevs_estim = counts
        prevs_estim = prevs_estim / prevs_estim.sum()
        
        adjusted_prevs = self.solve_adjustment(self.cond_prob_matrix, prevs_estim)

        prevalences = {_class:prevalence for _class,prevalence in zip(self.classes, adjusted_prevs)}
        
        return prevalences
    
    @classmethod
    def getCondProbMatrix(cls, classes, y, y_pred):
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
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
    def solve_adjustment(cls, PteCondEstim, prevs_estim):
        # solve for the linear system Ax = B with A=PteCondEstim and B = prevs_estim
        A = PteCondEstim
        B = prevs_estim
        try:
            adjusted_prevs = np.linalg.solve(A, B)
            adjusted_prevs = np.clip(adjusted_prevs, 0, 1)
            adjusted_prevs /= adjusted_prevs.sum()
        except np.linalg.LinAlgError:
            adjusted_prevs = prevs_estim  # no way to adjust them!
        except ValueError:
            adjusted_prevs = prevs_estim  # no way to adjust them!
        return adjusted_prevs
