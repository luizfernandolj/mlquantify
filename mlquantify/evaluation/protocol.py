from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Union, List, Tuple, Any
from sklearn.base import BaseEstimator
from time import time
from tqdm import tqdm

from ..methods import METHODS, AGGREGATIVE, NON_AGGREGATIVE
from ..utils.general import *
from ..utils.method import *
from . import MEASURES
from ..base import Quantifier

import mlquantify as mq

class Protocol(ABC):
    """Base class for evaluation protocols.
    
    Parameters
    ----------
    models : Union[List[Union[str, Quantifier]], str, Quantifier]
        List of quantification models, a single model name, or 'all' for all models.
    learner : BaseEstimator, optional
        Machine learning model to be used with the quantifiers. Required for model methods.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.
    random_state : int, optional
        Seed for random number generation. Default is 32.
    verbose : bool, optional
        Whether to print progress messages. Default is False.
    return_type : str, optional
        Type of return value ('predictions' or 'table'). Default is 'predictions'.
    measures : List[str], optional
        List of error measures to calculate. Must be in MEASURES or None. Default is None.
    columns : List[str], optional
        Columns to be included in the table. Default is ['ITERATION', 'QUANTIFIER', 'REAL_PREVS', 'PRED_PREVS', 'BATCH_SIZE'].
    
    Attributes
    ----------
    models : List[Quantifier]
        List of quantification models.
    learner : BaseEstimator
        Machine learning model to be used with the quantifiers.
    n_jobs : int
        Number of jobs to run in parallel.
    random_state : int
        Seed for random number generation.
    verbose : bool
        Whether to print progress messages.
    return_type : str
        Type of return value ('predictions' or 'table').
    measures : List[str]
        List of error measures to calculate.
    columns : List[str]
        Columns to be included in the table.
    
    Raises
    ------
    AssertionError
        If measures contain invalid error measures.
        If return_type is invalid.
        If columns does not contain ['QUANTIFIER', 'REAL_PREVS', 'PRED_PREVS'].
    
    Notes
    -----
    - The 'models' parameter can be a list of Quantifiers, a single Quantifier, a list of model names, a single model name, or 'all'.
    - If 'models' is a list of model names or 'all', 'learner' must be provided.
    - The 'all' option for 'models' will use all quantification models available in the library.
    - If 'models' is a Quantifier or list of Quantifier, 'learner' is not required. But the models must be initializated
    - You can pass your own model by passing a Quantifier object.
    - Columns must contain ['QUANTIFIER', 'REAL_PREVS', 'PRED_PREVS'].
    - If 'return_type' is 'table', the table will contain the columns specified in 'columns' and the error measures in 'measures'.
    - For creating your own protocol, you must have the attributes 'models', 'learner', 'n_jobs', 'random_state', 'verbose', 'return_type', 'measures', and 'columns'., but columns can be changed, as long as it contains ['QUANTIFIER', 'REAL_PREVS', 'PRED_PREVS'].
    
    See Also
    --------
    APP : Artificial Prevalence Protocol.
    NPP : Natural Prevalence Protocol.
    Quantifier : Base class for quantification methods.
    
    Examples
    --------
    import numpy as np
    >>> from mlquantify.evaluation.protocol import Protocol
    >>> from mlquantify.utils import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> import time as t
    >>> 
    >>> class MyProtocol(Protocol):
    ...    def __init__(self, 
    ...                models, 
    ...                learner, 
    ...                n_jobs, 
    ...                random_state, 
    ...                verbose, 
    ...                return_type, 
    ...                measures,
    ...                sample_size,
    ...                iterations=10):
    ...        super().__init__(models, 
    ...                         learner, 
    ...                         n_jobs, 
    ...                         random_state, 
    ...                         verbose, 
    ...                         return_type, 
    ...                         measures, 
    ...                         columns=['QUANTIFIER', 'REAL_PREVS', 'PRED_PREVS', 'TIME'])
    ...        self.sample_size = sample_size
    ...        self.iterations = iterations
    ...        
    ...    def predict_protocol(self, X_test, y_test):
    ...        predictions = []
    ...        
    ...        X_sample, y_sample = self._new_sample(X_test, y_test)
    ...        
    ...        for _ in range(self.iterations):
    ...            for model in self.models:
    ...                quantifier = model.__class__.__name__
    ...
    ...                real_prev = get_real_prev(y_sample)
    ...
    ...                start_time = t.time()
    ...                pred_prev = model.predict(X_sample)
    ...                end_time = t.time()
    ...                time = end_time - start_time
    ...                
    ...                predictions.append([quantifier, real_prev, pred_prev, time])
    ...        
    ...        return predictions
    ...    
    ...    def _new_sample(self, X_test, y_test):
    ...        indexes = np.random.choice(len(X_test), size=self.sample_size, replace=False)
    ...        return X_test[indexes], y_test[indexes]
    >>>     
    >>> 
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> 
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=42)
    >>> 
    >>> protocol = MyProtocol(models=["CC", "EMQ", "DyS"], # or [CC(learner), EMQ(learner), DyS(learner)]
    ...                    learner=RandomForestClassifier(),
    ...                    n_jobs=1,
    ...                    random_state=42,
    ...                    verbose=True,
    ...                    return_type="table",
    ...                    measures=None,
    ...                    sample_size=100)
    >>>
    >>> protocol.fit(X_train, y_train)
    >>> table = protocol.predict(X_test, y_test)
    >>> print(table)
    
    """
    
    def __init__(self,
                 models: Union[List[Union[str, Quantifier]], str, Quantifier],
                 learner: BaseEstimator = None,
                 n_jobs: int = 1,
                 random_state: int = 32,
                 verbose: bool = False,
                 return_type: str = "predictions",
                 measures: List[str] = None,
                 columns: List[str] = ["ITERATION", "QUANTIFIER", "REAL_PREVS", "PRED_PREVS", "BATCH_SIZE"]):
        
        assert not measures or all(m in MEASURES for m in measures), \
            f"Invalid measure(s) provided. Valid options: {list(MEASURES.keys())} or None"
        assert return_type in ["predictions", "table"], \
            "Invalid return_type. Valid options: ['predictions', 'table']"
        assert all(col in columns for col in ["QUANTIFIER", "REAL_PREVS", "PRED_PREVS"]), \
            "Columns must contain ['QUANTIFIER', 'REAL_PREVS', 'PRED_PREVS']"

        # Fixed parameters
        self.models = self._initialize_models(models, learner)
        self.learner = learner
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.return_type = return_type
        self.measures = measures
        self.columns = columns
        
    def _initialize_models(self, models, learner):
        """Initializes the quantification models.
        
        Parameters
        ----------
        models : Union[List[Union[str, Quantifier]], str, Quantifier]
            List of quantification models, a single model name, or 'all' for all models.
        learner : BaseEstimator
            Machine learning model to be used with the quantifiers.
        
        Returns
        -------
        List[Quantifier]
            List of quantification models.
        """
        if isinstance(models, list):
            if all(isinstance(model, Quantifier) for model in models):
                return models
            return [get_method(model)(learner) for model in models]
        
        if isinstance(models, Quantifier):
            return [models]

        assert learner is not None, "Learner is required for model methods."

        model_dict = {
            "all": METHODS.values,
            "aggregative": AGGREGATIVE.values,
            "non_aggregative": NON_AGGREGATIVE.values
        }

        if models in model_dict:
            return [model(learner) if hasattr(model, "learner") else model() for model in model_dict[models]()]
        return [get_method(models)(learner)]
    
    def sout(self, msg):
        """Prints a message if verbose is True."""
        if self.verbose:
            print('[APP]' + msg)
    
    def fit(self, X_train, y_train):
        """Fits the models with the training data.
        
        Parameters
        ----------
        X_train : np.ndarray
            Features of the training set.
        y_train : np.ndarray
            Labels of the training set.
        
        Returns
        -------
        Protocol
            Fitted protocol.
        """
        self.sout("Fitting models")

        args = ((model, X_train, y_train) for model in self.models)
        
        wrapper = tqdm if self.verbose else lambda x, **kwargs: x
    
        self.models = Parallel(n_jobs=self.n_jobs)(  # Parallel processing of models
            delayed(self._delayed_fit)(*arg) for arg in wrapper(args, desc="Fitting models", total=len(self.models))
        )
        self.sout("Fit [Done]")
        return self
    
    
    def predict(self, X_test: np.ndarray, y_test: np.ndarray) -> Any:
        """Predicts the prevalence for the test set.
        
        Parameters
        ----------
        X_test : np.ndarray
            Features of the test set.
        y_test : np.ndarray
            Labels of the test set.
        
        Returns
        -------
        Any
            Predictions for the test set. Can be a table or a tuple with the quantifier names, real prevalence, and predicted prevalence.
        """
        predictions = self.predict_protocol(X_test, y_test)
        predictions_df = pd.DataFrame(predictions, columns=self.columns)

        if self.return_type == "table":
            if self.measures:
                smoothed_factor = 1 / (2 * len(X_test))

                def smooth(values: np.ndarray) -> np.ndarray:
                    return (values + smoothed_factor) / (smoothed_factor * len(values) + 1)

                for metric in self.measures:
                    predictions_df[metric] = predictions_df.apply(
                        lambda row: get_measure(metric)( 
                            smooth(np.array(row["REAL_PREVS"])),
                            smooth(np.array(row["PRED_PREVS"]))
                        ),
                        axis=1
                    )
            return predictions_df

        return (
            predictions_df["QUANTIFIER"].to_numpy(),  # Quantifier names
            np.stack(predictions_df["REAL_PREVS"].to_numpy()),  # REAL_PREVS
            np.stack(predictions_df["PRED_PREVS"].to_numpy())   # PRED_PREVS
        )

    @abstractmethod
    def predict_protocol(self, X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        """Abstract method that every protocol must implement
        
        Parameters
        ----------
        X_test : np.ndarray
            Features of the test set.
        y_test : np.ndarray
            Labels of the test set.
        
        Returns
        -------
        np.ndarray
            Predictions for the test set. With the same format as the column names attribute.
        """
        ...

    @abstractmethod
    def _new_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        """Abstract method of sample extraction for each protocol.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing X_sample and y_sample.
        """
        ...

    @staticmethod
    def _delayed_fit(model, X_train, y_train):
        """Method to fit the model in parallel.
        
        Parameters
        ----------
        model : Quantifier
            Quantification model.
        X_train : np.ndarray
            Features of the training set.
        y_train : np.ndarray
            Labels of the training set.
        
        Returns
        -------
        Quantifier
            Fitted quantification model
        """
        model_name = model.__class__.__name__
        if model_name == "Ensemble" and isinstance(model.base_quantifier, Quantifier):
            model_name = f"{model.__class__.__name__}_{model.base_quantifier.__class__.__name__}_{model.size}"
        
        start = time()
        model = model.fit(X=X_train, y=y_train)
        duration = time() - start
        print(f"\tFitted {model_name} in {duration:.3f} seconds")
        return model


    
    
    


class APP(Protocol):
    """Artificial Prevalence Protocol. 
    
    This approach splits a test into several samples varying prevalence and sample size, 
    with n iterations. For a list of Quantifiers, it computes training and testing for 
    each one and returns either a table of results with error measures or just the predictions.
    
    Parameters
    ----------
    models : Union[List[Union[str, Quantifier]], str, Quantifier]
        List of quantification models, a single model name, or 'all' for all models.
    batch_size : Union[List[int], int]
        Size of the batches to be processed, or a list of sizes.
    learner : BaseEstimator, optional
        Machine learning model to be used with the quantifiers. Required for model methods.
    n_prevs : int, optional
        Number of prevalence points to generate. Default is 100.
    n_iterations : int, optional
        Number of iterations for the protocol. Default is 1.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.
    random_state : int, optional
        Seed for random number generation. Default is 32.
    verbose : bool, optional
        Whether to print progress messages. Default is False.
    return_type : str, optional
        Type of return value ('predictions' or 'table'). Default is 'predictions'.
    measures : List[str], optional
        List of error measures to calculate. Must be in MEASURES or None. Default is None.
    
    Attributes
    ----------
    models : List[Quantifier]
        List of quantification models.
    batch_size : Union[List[int], int]
        Size of the batches to be processed.
    learner : BaseEstimator
        Machine learning model to be used with the quantifiers.
    n_prevs : int
        Number of prevalence points to generate.
    n_iterations : int
        Number of iterations for the protocol.
    n_jobs : int
        Number of jobs to run in parallel.
    random_state : int
        Seed for random number generation.
    verbose : bool
        Whether to print progress messages.
    return_type : str
        Type of return value ('predictions' or 'table').
    measures : List[str]
        List of error measures to calculate.
        
    Raises
    ------
    AssertionError
        If return_type is invalid.
    
    See Also
    --------
    Protocol : Base class for evaluation protocols.
    NPP : Natural Prevalence Protocol.
    Quantifier : Base class for quantification methods.
    
    Examples
    --------
    >>> from mlquantify.evaluation.protocol import APP
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # Loading dataset from sklearn
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> 
    >>> #Splitting into train and test
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
    >>>
    >>> app = APP(models=["CC", "EMQ", "DyS"],
    ...           batch_size=[10, 50, 100],
    ...           learner=RandomForestClassifier(),
    ...           n_prevs=100, # Default
    ...           n_jobs=-1,
    ...           return_type="table",
    ...           measures=["ae", "se"],
    ...           verbose=True)
    >>>
    >>> app.fit(X_train, y_train)
    >>>
    >>> table = app.predict(X_test, y_test)
    >>>
    >>> print(table)
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
        
        super().__init__(models, learner, n_jobs, random_state, verbose, return_type, measures)
        self.n_prevs = n_prevs
        self.batch_size = batch_size if isinstance(batch_size, list) else [batch_size]
        self.n_prevs = n_prevs
        self.n_iterations = n_iterations


    def predict_protocol(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple:
        """Generates several samples with artificial prevalences and sizes.
        For each model, predicts with this sample, aggregating all together
        with a pandas dataframe if requested, or else just the predictions.

        Parameters
        ----------
        X_test : np.ndarray
            Features of the test set.
        y_test : np.ndarray
            Labels of the test set.
        
        Returns
        -------
        Tuple
            Tuple containing the (iteration, model name, prev, prev_pred, and batch size).
        """
        
        n_dim = len(np.unique(y_test))
        prevs = generate_artificial_prevalences(n_dim, self.n_prevs, self.n_iterations)

        args = [
            (iteration, X_test, y_test, model, prev, bs, self.verbose)
            for prev in prevs for bs in self.batch_size for model in self.models for iteration in range(self.n_iterations)
        ]
        
        size = len(prevs) * len(self.models) * len(self.batch_size) * self.n_iterations

        predictions = []
        for arg in tqdm(args, desc="Running APP", total=size):
            predictions.append(self._predict(*arg))
        
        return predictions

    def _predict(self, iteration:int, X: np.ndarray, y: np.ndarray, model: Any, prev: List[float], batch_size: int, verbose: bool) -> Tuple:
        """Method predicts into the new sample for each model and prevalence.

        Parameters
        ----------
        iteration : int
            Current iteration.
        X : np.ndarray
            Features of the test set.
        y : np.ndarray
            Labels of the test set.
        model : Any
            Quantification model.
        prev : List[float]
            Prevalence values for the sample.
        batch_size : int
            Batch size for the sample.
        verbose : bool
            Whether to print progress messages.
        
        Returns
        -------
        Tuple
            Tuple containing the iteration, model name, prev, prev_pred, and batch size.
        """
        model_name = model.__class__.__name__
        if model_name == "Ensemble" and isinstance(model.base_quantifier, Quantifier):
            model_name = f"{model.__class__.__name__}_{model.base_quantifier.__class__.__name__}_{model.size}"
        
        if verbose:
            print(f'\t {model_name} with {batch_size} instances and prev {prev}')
        
        X_sample, _ = self._new_sample(X, y, prev, batch_size)
        prev_pred = np.asarray(list(model.predict(X_sample).values()))
        
        if verbose:
            print(f'\t \\--Ending {model_name} with {batch_size} instances and prev {prev}\n')
        
        return (iteration+1, model_name, prev, prev_pred, batch_size)


    def _new_sample(self, X: np.ndarray, y: np.ndarray, prev: List[float], batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a new sample with a specified prevalence and size.

        Parameters
        ----------
        X : np.ndarray
            Features of the test set.
        y : np.ndarray
            Labels of the test set.
        prev : List[float]
            Prevalence values for the sample.
        batch_size : int
            Batch size for the sample.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing the new sample features and labels.
        """
        sample_index = generate_artificial_indexes(y, prev, batch_size, np.unique(y))
        return (np.take(X, sample_index, axis=0), np.take(y, sample_index, axis=0))
 











class NPP(Protocol):
    """Natural Prevalence Protocol.
    
    This approach splits a test into several samples varying sample size,
    with n iterations. For a list of Quantifiers, it computes training and testing for
    each one and returns either a table of results with error measures or just the predictions.
    
    Parameters
    ----------
    models : Union[List[Union[str, Quantifier]], str, Quantifier]
        List of quantification models, a single model name, or 'all' for all models.
    batch_size : Union[List[int], int]
        Size of the batches to be processed, or a list of sizes.
    learner : BaseEstimator, optional
        Machine learning model to be used with the quantifiers. Required for model methods.
    n_iterations : int, optional
        Number of iterations for the protocol. Default is 1.
    n_jobs : int, optional
        Number of jobs to run in parallel. Default is 1.
    random_state : int, optional
        Seed for random number generation. Default is 32.
    verbose : bool, optional
        Whether to print progress messages. Default is False.
    return_type : str, optional
        Type of return value ('predictions' or 'table'). Default is 'predictions'.
    measures : List[str], optional
        List of error measures to calculate. Must be in MEASURES or None. Default is None.
    
    Attributes
    ----------
    models : List[Quantifier]
        List of quantification models.
    batch_size : Union[List[int], int]
        Size of the batches to be processed.
    learner : BaseEstimator
        Machine learning model to be used with the quantifiers.
    n_iterations : int
        Number of iterations for the protocol.
    n_jobs : int
        Number of jobs to run in parallel.
    random_state : int
        Seed for random number generation.
    verbose : bool
        Whether to print progress messages.
    return_type : str
        Type of return value ('predictions' or 'table').
    measures : List[str]
        List of error measures to calculate.    
    """
    
    
    def __init__(self,     
                 models: Union[List[Union[str, Quantifier]], str, Quantifier], 
                 learner: BaseEstimator = None, 
                 n_jobs: int = 1,
                 random_state: int = 32,
                 verbose: bool = False,
                 return_type: str = "predictions",
                 measures: List[str] = None):
        
        super().__init__(models, learner, n_jobs, random_state, verbose, return_type, measures)
        
    
    def predict_protocol(self, X_test, y_test) -> tuple:
        raise NotImplementedError
        
    
    def _new_sample(self, X, y, prev: List[float], batch_size: int) -> tuple:
        raise NotImplementedError
        
    
    def _delayed_predict(self, args) -> tuple:
        raise NotImplementedError