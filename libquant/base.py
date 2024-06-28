from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator


class Quantifier(ABC, BaseEstimator):
    
    @abstractmethod
    def fit(self, *args, **kwargs):
        ...
      
    @abstractmethod  
    def estimate(self, *args, **kwargs):
        ...
