import numpy as np
import pandas as pd
from typing import Union , List
import itertools
from ...methods import get_class, METHODS
from ...utils import generate_indexes, parallel


class APP:
    
    def __init__(self, 
                 models:Union[List[str], str], 
                 batch_size:Union[List[int], int],
                 n_prevs:int=100,
                 n_iterations:int=1,
                 n_jobs=1,
                 random_state:int=32,
                 verbose:bool=False,
                 return_type:str="predictions"):
        
        assert return_type in ["predictions", "table"], "Return type option not valid, options are ['predictions', 'table']"
        assert all(model in METHODS for model in models), "Not all items in models are valid options"
        
        self.models = [get_class(model) for model in models] if isinstance(models, list) else [get_class(models)]
        self.batch_size = batch_size
        self.n_prevs = n_prevs
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
    
    def fit(self, X_train, y_train):
        self.models = parallel(_delayed_fit, self.models, self.n_jobs, X_train, y_train) 
        return self
        
        
    
    def predict(self, X_test, y_test):
        
        prevs = self.artificial_prevalence(np.unique(y_test), self.n_prevs, self.n_iterations)        
        
        predictions = np.array([])
        
        for model in self.models:
            args = ((X_test, y_test, model, prev) for prev in prevs)
            
            predictions_model = np.asarray(
                parallel(
                    _delayed_predict,
                    args,
                    n_jobs=self.n_jobs
                )
            )
            
            predictions = np.concatenate((predictions, predictions_model), axis=0)
        
        if self.return_type == "table":
            return pd.DataFrame(predictions)
        
        return (predictions[:, 0], predictions[:, 1], predictions[:, 2])
    
    
    def artificial_prevalence(self, n_dim:int, n_prev:int, n_iter:int):
        s = np.linspace(0., 1., n_prev, endpoint=True)
        s = [s] * (n_dim - 1)
        prevs = [p for p in itertools.product(*s, n_iter=1) if sum(p)<=1]
        
        prevs = [p+(1-sum(p),) for p in prevs]
        
        prevs = np.asarray(prevs).reshape(len(prevs), -1)
        if n_iter>1:
            prevs = np.repeat(prevs, n_iter, axis=0)
        return prevs
    

def _new_sample(X, y, prev:List[float], batch_size:int) -> np.ndarray:
    
    sample_index = generate_indexes(y, prev, batch_size, np.unique(y))
    
    X_sample = X[sample_index]
    y_sample = y[sample_index]
    
    return (X_sample, y_sample)

    
def _delayed_predict(args) -> tuple:
    X, y, model, prev, batch_size = args
    
    X_sample, y_sample = _new_sample(X, y, prev, batch_size)
    
    prev_pred = list(model.predict(X_sample, y_sample).values())
    
    return [model.__class__.__name__, prev, prev_pred, batch_size]
    


def _delayed_fit(model, X, y):
    return model.fit(X, y)
        
        
    