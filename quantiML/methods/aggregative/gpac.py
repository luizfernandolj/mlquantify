import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from ...base import AggregativeQuantifier
from sklearn.model_selection import StratifiedKFold
from .gac import GAC

class GPAC(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator):
        assert isinstance(learner, BaseEstimator), "learner object is not an estimator"
        self.learner = learner
        self.cond_prob_matrix = None
    
    def _fit_method(self, X, y, learner_fitted: bool = False, cv_folds: int = 10):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
            
        if learner_fitted:
            y_pred = self.learner.predict(X)
            y_label = y
        else:
            skf = StratifiedKFold(n_splits=cv_folds)    
            y_pred = []
            y_label = []
            
            for train_index, valid_index in skf.split(X,y):
                
                tr_data = pd.DataFrame(X.iloc[train_index])   #Train data and labels
                tr_label = y.iloc[train_index]
                
                valid_data = pd.DataFrame(X.iloc[valid_index])  #Validation data and labels
                valid_label = y.iloc[valid_index]
                
                self.learner.fit(tr_data, tr_label)
                
                y_pred.extend(self.learner.predict(valid_data))     #evaluating scores
                y_label.extend(valid_label)
        
        self.cond_prob_matrix = GAC.getCondProbMatrix(self.classes, y, y_pred)
        
        return self
    
    def _predict_method(self, X) -> dict:
        prevalences = {}
        
        y_pred = self.learner.predict(X)

        # Distribution of predictions in the test set
        prevs_estim = np.zeros(self.n_class)
        _, counts = np.unique(y_pred, return_counts=True)
        prevs_estim = counts
        prevs_estim = prevs_estim / prevs_estim.sum()
        
        adjusted_prevs = GAC.solve_adjustment(self.cond_prob_matrix, prevs_estim)

        prevalences = {_class:prevalence for _class,prevalence in zip(self.classes, adjusted_prevs)}
        
        return prevalences
    
    @classmethod
    def getPteCondEstim(cls, classes, y, y_pred):
        # estimate the matrix with entry (i,j) being the estimate of P(yi|yj), that is, the probability that a
        # document that belongs to yj ends up being classified as belonging to yi
        n_classes = len(classes)
        # confusion = np.zeros(shape=(n_classes, n_classes))
        CM = np.eye(n_classes)
        for i, class_ in enumerate(classes):
            idx = y == class_
            if idx.any():
                CM[i] = y_pred[idx].mean(axis=0)
        return CM.T