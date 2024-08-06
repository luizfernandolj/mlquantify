import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.base import BaseEstimator
import itertools
from ...utils import generate_indexes, parallel, get_real_prev
from ..measures import get_measure, MEASURES

class APP:
    
    def __init__(self,
                 learner: BaseEstimator, 
                 models:Union[List[str], str], 
                 batch_size:Union[List[int], int],
                 n_prevs:int=100,
                 n_iterations:int=1,
                 n_jobs=1,
                 random_state:int=32,
                 verbose:bool=False,
                 return_type:str="predictions",
                 measures:List[str]=None):
        
        from ...methods import get_method
        
        assert all(m in MEASURES for m in measures) or not measures, f"Not valid option for measures, options are: {list(MEASURES.keys())} or None"
        assert return_type in ["predictions", "table"], "Return type option not valid, options are ['predictions', 'table']"
        
        self.learner = learner
        self.models = [get_method(model)(learner) for model in models] if isinstance(models, list) else [get_method(models)(learner),]
        self.batch_size = batch_size
        self.n_prevs = n_prevs
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.return_type = return_type
        self.measures = measures
        
    def fit(self, X_train, y_train):
        args = ((model, X_train, y_train) for model in self.models)
        self.models = parallel(_delayed_fit, args, self.n_jobs)
        return self
        
    def predict(self, X_test, y_test):
        n_dim = len(np.unique(y_test))
        prevs = self.artificial_prevalence(n_dim, self.n_prevs, self.n_iterations)

        predictions = []

        for model in self.models:
            if isinstance(self.batch_size, list):
                args = ((X_test, y_test, model, prev, bs) for prev in prevs for bs in self.batch_size)
            else:
                args = args = ((X_test, y_test, model, prev, self.batch_size) for prev in prevs)
            
            predictions_model = parallel(
                _delayed_predict,
                args,
                n_jobs=self.n_jobs
            )  

            predictions.extend(predictions_model)

        predictions_df = pd.DataFrame(predictions)
        
        if self.return_type == "table":
            predictions_df.columns = ["QUANTIFIER", "REAL_PREVS", "PRED_PREVS", "BATCH_SIZE"]
            
            if self.measures:
                
                def smooth(values:np.ndarray) ->np.ndarray:
                    smoothed_factor = 1/(2 * len(X_test))
                    
                    f1 = (smoothed_factor + values)
                    f2 = (smoothed_factor * len(y_test) + sum(values))

                    return f1 / f2
                
                
                for metric in self.measures:
                    predictions_df[metric] = predictions_df.apply(
                        lambda row: get_measure(metric)(smooth(row["REAL_PREVS"]), smooth(row["PRED_PREVS"])),
                        axis=1
                    )
            
            return predictions_df
        
        predictions_array = predictions_df.to_numpy()
        return (
            predictions_array[:, 0],  # Model names
            np.stack(predictions_array[:, 1]),  # Prev
            np.stack(predictions_array[:, 2])   # Prev_pred
        )  # MODEL, REAL PREV, PRED PREV
    
    def artificial_prevalence(self, n_dim:int, n_prev:int, n_iter:int):
        s = np.linspace(0., 1., n_prev, endpoint=True)
        s = [s] * (n_dim - 1)
        prevs = [p for p in itertools.product(*s, repeat=1) if sum(p)<=1]
        
        prevs = [p+(1-sum(p),) for p in prevs]
        
        prevs = np.asarray(prevs).reshape(len(prevs), -1)
        if n_iter > 1:
            prevs = np.repeat(prevs, n_iter, axis=0)
        return prevs

def _new_sample(X, y, prev:List[float], batch_size:int) -> tuple:
    sample_index = generate_indexes(y, prev, batch_size, np.unique(y))
    X_sample = X[sample_index]
    y_sample = y[sample_index]
    return (X_sample, y_sample)

def _delayed_predict(args) -> tuple:
    X, y, model, prev, batch_size = args
    X_sample, _ = _new_sample(X, y, prev, batch_size)
    prev_pred = np.asarray(list(model.predict(X=X_sample).values()))
    return [model.__class__.__name__, prev, prev_pred, batch_size]

def _delayed_fit(args):
    model, X_train, y_train = args
    return model.fit(X=X_train, y=y_train)
