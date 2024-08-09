from .base import Quantifier
from typing import Union, List
import itertools
from tqdm import tqdm
import signal
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
from .utils import parallel
from .evaluation import get_measure, APP, NPP

class GridSearchQ(Quantifier):
    """
    Hyperparameter optimization for quantification models using grid search.

    Args:
        model (Quantifier): The base quantification model.
        param_grid (dict): Hyperparameters to search over.
        protocol (str, optional): Quantification protocol ('app' or 'npp'). Defaults to 'app'.
        n_prevs (int, optional): Number of prevalence points for APP. Defaults to None.
        n_repetitions (int, optional): Number of repetitions for NPP. Defaults to 1.
        scoring (Union[List[str], str], optional): Metric(s) for evaluation. Defaults to "mae".
        refit (bool, optional): Refit model on best parameters. Defaults to True.
        val_split (float, optional): Proportion of data for validation. Defaults to 0.4.
        n_jobs (int, optional): Number of parallel jobs. Defaults to 1.
        random_seed (int, optional): Seed for reproducibility. Defaults to 42.
        timeout (int, optional): Max time per parameter combination (seconds). Defaults to -1.
        verbose (bool, optional): Verbosity of output. Defaults to False.
    """

    def __init__(self,
                 model: Quantifier,
                 param_grid: dict,
                 protocol: str = 'app',
                 n_prevs: int = None,
                 n_repetitions: int = 1,
                 scoring: Union[List[str], str] = "ae",
                 refit: bool = True,
                 val_split: float = 0.4,
                 n_jobs: int = 1,
                 random_seed: int = 42,
                 timeout: int = -1,
                 verbose: bool = False):
        
        self.model = model
        self.param_grid = param_grid
        self.protocol = protocol.lower()
        self.n_prevs = n_prevs
        self.n_repetitions = n_repetitions
        self.refit = refit
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.timeout = timeout
        self.verbose = verbose
        self.scoring = [get_measure(measure) for measure in (scoring if isinstance(scoring, list) else [scoring])]

        assert self.protocol in {'app', 'npp'}, 'Unknown protocol; valid ones are "app" or "npp".'
        
        if self.protocol == 'npp' and self.n_repetitions <= 1:
            raise ValueError('For "npp" protocol, n_repetitions must be greater than 1.')

    def sout(self, msg):
        """Prints messages if verbose is True."""
        if self.verbose:
            print(f'[{self.__class__.__name__}]: {msg}')

    def __get_protocol(self, model, sample_size):
        """Get the appropriate protocol instance.

        Args:
            model (Quantifier): The quantification model.
            sample_size (int): The sample size for batch processing.

        Returns:
            object: Instance of APP or NPP protocol.
        """
        protocol_params = {
            'models': model,
            'batch_size': sample_size,
            'n_iterations': self.n_repetitions,
            'n_jobs': self.n_jobs,
            'verbose': False,
            'random_state': 35,
            'return_type': "predictions"
        }
        return APP(n_prevs=self.n_prevs, **protocol_params) if self.protocol == 'app' else NPP(**protocol_params)

    def fit(self, X, y):
        """Fit the quantifier model and perform grid search.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels.

        Returns:
            self: Fitted GridSearchQ instance.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, random_state=self.random_seed)
        param_combinations = list(itertools.product(*self.param_grid.values()))
        best_score, best_params = None, None
        
        if self.timeout > 0:
            signal.signal(signal.SIGALRM, self._timeout_handler)

        def evaluate_combination(params):
            """Evaluate a single combination of hyperparameters.

            Args:
                params (tuple): A tuple of hyperparameter values.

            Returns:
                float or None: The evaluation score, or None if timeout occurred.
            """
            
            if self.verbose:
                print(f"\tEvaluate Combination for {str(params)}")
            
            
            model = deepcopy(self.model)
            model.set_params(**dict(zip(self.param_grid.keys(), params)))
            protocol_instance = self.__get_protocol(model, len(y_train))

            try:
                if self.timeout > 0:
                    signal.alarm(self.timeout)

                protocol_instance.fit(X_train, y_train)
                _, real_prevs, pred_prevs = protocol_instance.predict(X_val, y_val)
                scores = [np.mean([measure(rp, pp) for rp, pp in zip(real_prevs, pred_prevs)]) for measure in self.scoring]

                if self.timeout > 0:
                    signal.alarm(0)
                    
                    
                    
                if self.verbose:
                    print(f"\t\\--ended evaluation of {str(params)}")

                return np.mean(scores) if scores else None
            except TimeoutError:
                self.sout(f'Timeout reached for combination {params}.')
                return None

        results = parallel(
            evaluate_combination,
            tqdm(param_combinations, desc="Evaluating combination", total=len(param_combinations)) if self.verbose else param_combinations,
            n_jobs=self.n_jobs
        )
        
        for score, params in zip(results, param_combinations):
            if score is not None and (best_score is None or score < best_score):
                best_score, best_params = score, params

        self.best_score_ = best_score
        self.best_params_ = dict(zip(self.param_grid.keys(), best_params))
        self.sout(f'Optimization complete. Best score: {self.best_score_}, with parameters: {self.best_params_}.')

        if self.refit and self.best_params_:
            self.model.set_params(**self.best_params_)
            self.model.fit(X, y)
            self.best_model_ = self.model

        return self
    
    def predict(self, X):
        """Make predictions using the best found model.

        Args:
            X (array-like): Data to predict on.

        Returns:
            array-like: Predictions.
        """
        if not hasattr(self, 'best_model_'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.best_model_.predict(X)
    
    @property
    def classes_(self):
        """Get the classes of the best model.

        Returns:
            array-like: The classes.
        """
        return self.best_model_.classes_

    def set_params(self, **parameters):
        """Set the hyperparameters for grid search.

        Args:
            parameters (dict): Hyperparameters to set.
        """
        self.param_grid = parameters

    def get_params(self, deep=True):
        """Get the parameters of the best model.

        Args:
            deep (bool, optional): If True, will return the parameters for this estimator and contained subobjects. Defaults to True.

        Returns:
            dict: Parameters of the best model.
        """
        if hasattr(self, 'best_model_'):
            return self.best_model_.get_params()
        raise ValueError('get_params called before fit')

    def best_model(self):
        """Return the best model after fitting.

        Returns:
            Quantifier: The best model.

        Raises:
            ValueError: If called before fitting.
        """
        if hasattr(self, 'best_model_'):
            return self.best_model_
        raise ValueError('best_model called before fit')

    def _timeout_handler(self, signum, frame):
        """Handle timeouts during evaluation.

        Args:
            signum (int): Signal number.
            frame (object): Current stack frame.
        
        Raises:
            TimeoutError: When the timeout is reached.
        """
        raise TimeoutError()
