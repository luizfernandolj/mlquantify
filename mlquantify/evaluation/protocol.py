from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, List
import itertools
from sklearn.base import BaseEstimator
from time import time
from tqdm import tqdm

from ..methods import METHODS, AGGREGATIVE, NON_AGGREGATIVE
from ..utils.general import *
from ..utils.method import *
from . import MEASURES
from ..base import Quantifier

class Protocol(ABC):
    """Base class for implementing different quantification protocols.

    This abstract class provides a structure for creating protocols that involve
    fitting quantification models to training data and generating predictions on test data.
    It supports parallel processing, multiple iterations, and different output formats.

    Args:
        models (Union[List[Union[str, Quantifier]], str, Quantifier]): 
            List of quantification models, a single model name, or 'all' for all models.
        batch_size (Union[List[int], int]): 
            Size of the batches to be processed, or a list of sizes.
        learner (BaseEstimator, optional): 
            Machine learning model to be used with the quantifiers. Required for model methods.
        n_iterations (int, optional): 
            Number of iterations for the protocol. Default is 1.
        n_jobs (int, optional): 
            Number of jobs to run in parallel. Default is 1.
        random_state (int, optional): 
            Seed for random number generation. Default is 32.
        verbose (bool, optional): 
            Whether to print progress messages. Default is False.
        return_type (str, optional): 
            Type of return value ('predictions' or 'table'). Default is 'predictions'.
        measures (List[str], optional): 
            List of error measures to calculate. Must be in MEASURES or None. Default is None.
    """


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

        assert not measures or all(m in MEASURES for m in measures), \
            f"Invalid measure(s) provided. Valid options: {list(MEASURES.keys())} or None"
        assert return_type in ["predictions", "table"], \
            "Invalid return_type. Valid options: ['predictions', 'table']"

        self.models = self._initialize_models(models, learner)
        self.learner = learner
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.return_type = return_type
        self.measures = measures

    def _initialize_models(self, models, learner):
        if isinstance(models, list):
            if isinstance(models[0], Quantifier):
                return models
            assert learner is not None, "Learner is required for model methods."
            return [get_method(model)(learner) for model in models]
        if isinstance(models, Quantifier):
            return [models]
        
        assert learner is not None, "Learner is required for model methods."
        
        if models == "all":
            print(hasattr(list(AGGREGATIVE.values())[0], "learner"))  
            models = [model(learner) if hasattr(model, "learner") else model() for model in METHODS.values()]
            return models
        if models == "aggregative":
            return [model(learner) for model in AGGREGATIVE.values()]
        if models == "non_aggregative":
            return [model() for model in NON_AGGREGATIVE.values()]
            
        return [get_method(models)(learner)]
    
    
    def sout(self, msg):
        if self.verbose:
            print('[APP]' + msg)    
    
        
    def fit(self, X_train, y_train):
        """Fit all methods into the training data.

        Args:
            X_train (array-like): Features of training.
            y_train (array-like): Labels of training.
        """
        self.sout("Fitting models")

        args = ((model, X_train, y_train, self.verbose) for model in self.models)
        self.models = parallel(
            self._delayed_fit, 
            tqdm(args, desc="Fitting models", total=len(self.models)) if self.verbose else args, 
            self.n_jobs)
        
        self.sout("Fit [Done]")
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
        
        
        predictions = self.predict_protocol(X_test, y_test) 


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
        
        
    @abstractmethod
    def predict_protocol(self) -> np.ndarray:
        """ Abstract method that every protocol has to implement """
        ...
        
    @abstractmethod
    def _new_sample(self) -> tuple:
        """ Abstract method of sample extraction for each protocol

        Returns:
            tuple: tuple containing the X_sample and the y_sample
        """
        ...
        
        
    @abstractmethod
    def _delayed_predict(self, args) -> tuple:
        """abstract method for predicting in the extracted
        samples, is delayed for running in parallel for 
        eficciency purposes.
        """
        ...
        

    
    def _delayed_fit(self, args):
        model, X_train, y_train, verbose = args
        
        if verbose:
            print(f"\tFitting {model.__class__.__name__}")
            start = time()
        
        model = model.fit(X=X_train, y=y_train)
        
        if verbose:
            end = time()
            print(f"\t\\--Fit ended for {model.__class__.__name__} in {round(end - start, 3)} seconds")
        return model
    
    
    
    
    
    
    
    


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
        raise NotImplementedError
        
    
    def _new_sample(self, X, y, prev: List[float], batch_size: int) -> tuple:
        raise NotImplementedError
        
    
    def _delayed_predict(self, args) -> tuple:
        raise NotImplementedError