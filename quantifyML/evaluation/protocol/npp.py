import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.base import BaseEstimator

from ...base import Quantifier
from ._Protocol import Protocol


class NPP(Protocol):
    
    
    def __init__(self,     
                 models: Union[List[Union[str, Quantifier]], str, Quantifier], 
                 batch_size: Union[List[int], int],
                 learner: BaseEstimator = None, 
                 n_iterations: int = 1,
                 n_jobs: int = 1,
                 random_state: int = 32,
                 verbose: bool = False,
                 return_type: str = "predictions",
                 measures: List[str] = None):
        
        super().__init__(models, batch_size, learner, n_iterations, n_jobs, random_state, verbose, return_type, measures)
        
    
    def predict_protocol(self, X_test, y_test) -> tuple:
        
        ...
        
    
    def _new_sample(self, X, y, prev: List[float], batch_size: int) -> tuple:
        ...
        
    
    def _delayed_predict(self, args) -> tuple:
        ...