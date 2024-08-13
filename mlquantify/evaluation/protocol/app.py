import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.base import BaseEstimator
import itertools
from tqdm import tqdm

from ...utils import generate_artificial_indexes, parallel
from ...base import Quantifier
from ._Protocol import Protocol

class APP(Protocol):
    """Artificial Prevalence Protocol. It splits a test into several
    samples varying prevalence and sample size, with n iterations.
    For a list of Quantifiers, it computes training and testing 
    for each one and returns either a table of results with error measures
    or just the predictions.
    """
    
    def __init__(self,     
                 models: Union[List[Union[str, Quantifier]], str, Quantifier], 
                 batch_size: Union[List[int], int],
                 learner: BaseEstimator = None, 
                 n_prevs: int = 100,
                 n_iterations: int = 1,
                 n_jobs: int = 1,
                 random_state: int = 32,
                 verbose: bool = False,
                 return_type: str = "predictions",
                 measures: List[str] = None):
        
        super().__init__(models, batch_size, learner, n_iterations, n_jobs, random_state, verbose, return_type, measures)
        self.n_prevs = n_prevs

    def predict_protocol(self, X_test, y_test) -> tuple:
        """Generates several samples with artificial prevalences and sizes.
        For each model, predicts with this sample, aggregating all together
        with a pandas dataframe if requested, or else just the predictions.

        Args:
            X_test (array-like): Features of the test set.
            y_test (array-like): Labels of the test set.

        Returns:
            tuple: predictions containing the model name, real prev, pred prev, and batch size
        """
        
        n_dim = len(np.unique(y_test))
        prevs = self._generate_artificial_prevalences(n_dim, self.n_prevs, self.n_iterations)

        args = self._generate_args(X_test, y_test, prevs)
        batch_size = 1
        
        if isinstance(self.batch_size, list):
            batch_size = len(self.batch_size)
        
        size = len(prevs) * len(self.models) * batch_size * self.n_iterations
        
        predictions = parallel(
            self._delayed_predict,
            tqdm(args, desc="Running APP", total=size) if self.verbose else args,
            n_jobs=self.n_jobs
        )
        
        return predictions


    def _new_sample(self, X, y, prev: List[float], batch_size: int) -> tuple:
        """Generates a new sample with a specified prevalence and size.

        Args:
            X (array-like): Features from which to take the new sample.
            y (array-like): Labels from which to take the new sample.
            prev (List[float]): The specified prevalences.
            batch_size (int): Sample size.

        Returns:
            tuple: New sample's features and labels.
        """
        sample_index = generate_artificial_indexes(y, prev, batch_size, np.unique(y))
        return np.take(X, sample_index, axis=0), np.take(y, sample_index, axis=0)



    def _delayed_predict(self, args) -> tuple:
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
        
        X_sample, _ = self._new_sample(X, y, prev, batch_size)
        prev_pred = np.asarray(list(model.predict(X=X_sample).values()))
        
        if verbose:
            print(f'\t \\--Ending {model.__class__.__name__} with {str(batch_size)} instances and prev {str(prev)} \n')
        
        return [model.__class__.__name__, prev, prev_pred, batch_size]
    
    


    def _generate_artificial_prevalences(self, n_dim: int, n_prev: int, n_iter: int) -> np.ndarray:
        """Generates n artificial prevalences with n dimensions.

        Args:
            n_dim (int): Number of dimensions for the artificial prevalence.
            n_prev (int): Number of prevalence points to generate.
            n_iter (int): Number of iterations.

        Returns:
            np.ndarray: Generated artificial prevalences.
        """
        s = np.linspace(0., 1., n_prev, endpoint=True)
        prevs = np.array([p + (1 - sum(p),) for p in itertools.product(*(s,) * (n_dim - 1)) if sum(p) <= 1])
        
        return np.repeat(prevs, n_iter, axis=0) if n_iter > 1 else prevs



    def _generate_args(self, X_test, y_test, prevs):
        """Generates arguments for parallel processing based on the model, prevalence, and batch size.

        Args:
            X_test (array-like): Features of the test set.
            y_test (array-like): Labels of the test set.
            prevs (np.ndarray): Artificial prevalences generated.

        Returns:
            List[tuple]: List of arguments for parallel processing.
        """
        if isinstance(self.batch_size, list):
            return [(X_test, y_test, model, prev, bs, self.verbose) 
                    for prev in prevs for bs in self.batch_size for model in self.models]
        else:
            return [(X_test, y_test, model, prev, self.batch_size, self.verbose) 
                    for prev in prevs for model in self.models]
