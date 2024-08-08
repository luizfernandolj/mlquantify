from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, List
from sklearn.base import BaseEstimator
from time import time
from tqdm import tqdm

from ...methods import get_method, METHODS, AGGREGATIVE, NON_AGGREGATIVE
from ...utils import *
from ..measures import get_measure, MEASURES
from ...base import Quantifier, AggregativeQuantifier

class Protocol(ABC):

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