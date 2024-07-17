from abc import abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ....base import AggregativeQuantifier
from ....utils.utilities import GetScores
from ....utils.distances import probsymm, sqEuclidean, topsoe, hellinger

class MixtureModel(AggregativeQuantifier):
    
    def __init__(self, learner: BaseEstimator, measure:str=None):
        assert measure in ["hellinger", "topsoe", "probsymm", None], "measure not valid, please choose between: hellinger, topsoe or probsymm or None if it is a sample operation with scores"
        self.learner = learner
        self.measure = measure
        self.pos_scores = None
        self.neg_scores = None
    
    @property
    def multiclass_method(self) -> bool:
        return False
    
        
    def _fit_method(self, X, y, learner_fitted:bool=False, cv_folds:int=10):
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
            
        if learner_fitted or cv_folds is None:
            probabilities = self.learner.predict_proba(X)[:, 1]
            y_label = y
        else:   
            y_label, probabilities = GetScores(X, y, self.learner, cv_folds, learner_fitted)
            
        self.pos_scores = probabilities[y_label == self.classes[1]]
        self.neg_scores = probabilities[y_label == self.classes[0]]
        
        self.learner.fit(X, y)
        
        return self
    
    def _predict_method(self, X) -> dict:
        prevalences = {}
        
        test_scores = self.learner.predict_proba(X)[:, 1]
        
        prevalence = self._compute_prevalence(self.pos_scores, self.neg_scores, test_scores, self.measure)
        
        prevalences[self.classes[1]] = prevalence
        prevalences[self.classes[0]] = 1 - prevalence
        
        return prevalences
    
    @abstractmethod
    def _compute_prevalence(self, pos_scores:np.ndarray, neg_scores:np.ndarray, test_scores:np.ndarray, measure:str) -> float:
        ...
        
    def get_distance(self, dist_train, dist_test, measure):
        """This function applies a selected distance metric"""
        
        if sum(dist_train)<1e-20 or sum(dist_test)<1e-20:
            raise "One or both vector are zero (empty)..."
        if len(dist_train)!=len(dist_test):
            raise "Arrays need to be of equal sizes..."
        
        #use numpy arrays for efficient coding
        dist_train=np.array(dist_train,dtype=float);dist_test=np.array(dist_test,dtype=float)
        #Correct for zero values
        dist_train[np.where(dist_train<1e-20)]=1e-20
        dist_test[np.where(dist_test<1e-20)]=1e-20
        
        
        if measure == 'topsoe':
            return topsoe(dist_train, dist_test)
        if measure == 'probsymm':
            return probsymm(dist_train, dist_test)
        if measure == 'hellinger':
            return hellinger(dist_train, dist_test)
        return 100
    