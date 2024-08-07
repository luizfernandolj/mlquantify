import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.base import BaseEstimator
import itertools
from tqdm import tqdm
from ...utils import generate_indexes, parallel
from ..measures import get_measure, MEASURES
from ...base import Quantifier

class APP:
    """Artificial Prevalence Protocol, it split a test into several
    samples varying prevalence and the sample size, with n iterations,
    And, give a list of Quantifiers, computes the training and testing 
    for each one and can return a table of results with error measures
    if requested, or just the predictions.
    """
    
    
    def __init__(self,     
                 models:Union[List[str], str, List[Quantifier], Quantifier], 
                 batch_size:Union[List[int], int],
                 learner: BaseEstimator = None, 
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
        
        if isinstance(models, list):
            if isinstance(models[0], Quantifier):
                self.models = models
            else:
                assert learner, "Learner for the methods not passed"
                self.models = [get_method(model)(learner) for model in models]
        else:
            if isinstance(models, Quantifier):
                self.models = [models]
            else:
                assert learner, "Learner for the methods not passed"
                self.models = [get_method(models)(learner)]
            
        self.learner = learner
        self.batch_size = batch_size
        self.n_prevs = n_prevs
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.return_type = return_type
        self.measures = measures
        
    def fit(self, X_train, y_train):
        """Fit all methods into the training data.

        Args:
            X_train (array-like): Features of training.
            y_train (array-like): Labels of training.
        """
        
        args = ((model, X_train, y_train) for model in self.models)
        self.models = parallel(_delayed_fit, args, self.n_jobs)
        return self
        
    def predict(self, X_test, y_test) -> np.any:
        """Generate several samples with artificial prevalences, and sizes. 
        And for each method, predicts with this sample, aggregating all toguether
        with a pandas dataframe if request, or else just the predictions.

        Args:
            X_test (array-like): Features of test.
            y_test (array-like): Labels of test.

        Returns:
            tuple: tuple containing the model, real_prev and pred_prev, or.
            DataFrame: table of results, along with error measures if requested. 
        """
        
        
        n_dim = len(np.unique(y_test))
        prevs = self.artificial_prevalence(n_dim, self.n_prevs, self.n_iterations)

        predictions = []

        for model in self.models:
            if isinstance(self.batch_size, list):
                args = ((X_test, y_test, model, prev, bs, self.verbose) for prev in prevs for bs in self.batch_size)
                size = len(prevs) * len(self.batch_size)
            else:
                args = args = ((X_test, y_test, model, prev, self.batch_size, self.verbose) for prev in prevs)
                size = len(prevs)
            
            predictions_model = parallel(
                _delayed_predict,
                tqdm(args, desc="Running APP", total=size) if self.verbose else args,
                n_jobs=self.n_jobs
            )  

            predictions.extend(predictions_model)

        predictions_df = pd.DataFrame(predictions)
        
        if self.return_type == "table":
            predictions_df.columns = ["QUANTIFIER", "REAL_PREVS", "PRED_PREVS", "BATCH_SIZE"]
            
            if self.measures:
                
                def smooth(values:np.ndarray) ->np.ndarray:
                    smoothed_factor = 1/(2 * len(X_test))
                    
                    values = (values + smoothed_factor) / (smoothed_factor * len(values) + 1)

                    return values
                
                
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
        )
    
    
    
    def artificial_prevalence(self, n_dim:int, n_prev:int, n_iter:int) -> np.ndarray:
        """Generates n artificial prevalences with n_dimensions 

        Args:
            n_dim (int): n_dimensions of the artificial prevalence
            n_prev (int): number of prevs
            n_iter (int): number of iterations

        Returns:
            np.ndarray: _description_
        """
        s = np.linspace(0., 1., n_prev, endpoint=True)
        s = [s] * (n_dim - 1)
        prevs = [p for p in itertools.product(*s, repeat=1) if sum(p)<=1]
        
        prevs = [p+(1-sum(p),) for p in prevs]
        
        prevs = np.asarray(prevs).reshape(len(prevs), -1)
        if n_iter > 1:
            prevs = np.repeat(prevs, n_iter, axis=0)
        return prevs

def _new_sample(X, y, prev:List[float], batch_size:int) -> tuple:
    """Generate a new sample from wich it has a specift prevalence and size.

    Args:
        X (array-like): Features of test from where to take the new sample
        y (_type_): Labels of test from where to take the new sample
        prev (List[float]): the specific prevalences
        batch_size (int): sample size

    Returns:
        tuple: New sample's features and labels.
    """
    sample_index = generate_indexes(y, prev, batch_size, np.unique(y))
    X_sample = X[sample_index]
    y_sample = y[sample_index]
    return (X_sample, y_sample)


def _delayed_predict(args) -> tuple:
    """Method predicts into the new sample, is delayed for running 
    in parallel for eficciency purposes

    Args:
        args (Any): arguments to use 

    Returns:
        tuple: returns the (method name, real_prev, pred_prev and sample_size)
    """
    
    X, y, model, prev, batch_size, verbose = args
    
    if verbose:
        print(f'\t {model.__class__.__name__} with {str(batch_size)} instances and prev {str(prev)}')
    
    X_sample, _ = _new_sample(X, y, prev, batch_size)
    prev_pred = np.asarray(list(model.predict(X=X_sample).values()))
    
    if verbose:
        print(f'\t \\--Ending {model.__class__.__name__} with {str(batch_size)} instances and prev {str(prev)} \n')
    
    return [model.__class__.__name__, prev, prev_pred, batch_size]

def _delayed_fit(args):
    model, X_train, y_train = args
    return model.fit(X=X_train, y=y_train)
