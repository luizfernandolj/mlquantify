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
    GridSearchQ performs hyperparameter optimization for quantification models.
    
    This class takes a quantification model and a grid of hyperparameters, then evaluates 
    all combinations of these hyperparameters using a single train/validation split.
    It supports different quantification protocols like APP (Artificial Prevalence Protocol) 
    and NPP (Natural Prevalence Protocol), and can handle multiple scoring metrics.

    Attributes:
    -----------
    model : Quantifier
        The base quantification model to be optimized.
        
    param_grid : dict
        Dictionary where keys are hyperparameter names and values are lists of parameter settings to try.
        
    protocol : str, default='app'
        The quantification protocol to use, either 'app' for Artificial Prevalence Protocol 
        or 'npp' for Natural Prevalence Protocol.
        
    n_prevs : int, optional
        Number of prevalence points to use in the APP protocol. Ignored if using NPP.
        
    n_repetitions : int, default=1
        Number of repetitions for the NPP protocol. Must be greater than 1 if protocol is 'npp'.
        
    scoring : Union[List[str], str], default="mae"
        The scoring metric(s) to use for evaluation. Can be a string or a list of strings. 
        Each string should correspond to a valid metric name.
        
    refit : bool, default=True
        If True, refit the model with the best found parameters on the whole dataset.
        
    val_split : float, default=0.4
        The proportion of the data to use for validation.
        
    n_jobs : int, default=1
        Number of jobs to run in parallel during the grid search.
        
    random_seed : int, default=42
        The random seed for reproducibility.
        
    timeout : int, default=-1
        Maximum time in seconds allowed for each parameter combination evaluation. 
        If -1, no timeout is applied.
        
    verbose : bool, default=False
        If True, prints progress messages during the grid search.

    Methods:
    --------
    fit(X, y):
        Fits the quantification model using the provided dataset and performs grid search 
        to find the best hyperparameters.
        
    predict(X):
        Makes predictions using the best found model.
        
    classes_:
        Returns the classes of the best found model.
        
    set_params(**parameters):
        Sets the hyperparameters for the grid search.
        
    get_params(deep=True):
        Returns the parameters of the best found model after fitting.
        
    best_model():
        Returns the best model after fitting.
    """

    def __init__(self,
                 model: Quantifier,
                 param_grid: dict,
                 protocol: str = 'app',
                 n_prevs: int = None,
                 n_repetitions: int = 1,
                 scoring: Union[List[str], str] = "mae",
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

        assert self.protocol in {'app', 'npp'}, \
            'Unknown protocol; valid ones are "app" or "npp".'
        
        if self.protocol == 'npp':
            if not self.n_repetitions > 1:
                raise ValueError('For "npp" protocol, n_repetitions must be greater than 1.')

    def sout(self, msg):
        if self.verbose:
            print(f'[{self.__class__.__name__}]: {msg}')
    
    def __get_protocol(self, model, sample_size):
        commons = {
            'models': model,
            'batch_size': sample_size,
            'n_iterations': self.n_repetitions,
            'n_jobs': self.n_jobs,
            'verbose': False,
            'random_state': 35,
            'return_type': "predictions"
        }
        if self.protocol == 'app':
            return APP(n_prevs=self.n_prevs, **commons)
        elif self.protocol == 'npp':
            return NPP(**commons)
        else:
            raise ValueError('Unknown protocol.')
    
    def fit(self, X, y):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, random_state=self.random_seed)
        param_combinations = list(itertools.product(*self.param_grid.values()))
        best_score = None
        best_params = None
        some_timeouts = False
        
        if self.timeout > 0:
            def handler(signum, frame):
                self.sout('Timeout reached.')
                raise TimeoutError()
            signal.signal(signal.SIGALRM, handler)

        def evaluate_combination(params):
            model_params = dict(zip(self.param_grid.keys(), params))
            
            if self.verbose:
                print(f"\tEvaluate Combination for {str(params)}")
            
            model = deepcopy(self.model)
            model.set_params(**model_params)
            protocol_instance = self.__get_protocol(model, len(y_train))
            scores = []
            
            try:
                if self.timeout > 0:
                    signal.alarm(self.timeout)
                protocol_instance.fit(X_train, y_train)
                _, real_prevs, pred_prevs = protocol_instance.predict(X_val, y_val)
                
                # Iterar sobre real_prevs e pred_prevs para calcular a métrica
                scores = [np.mean([measure(rp, pp) for rp, pp in zip(real_prevs, pred_prevs)]) for measure in self.scoring]
                
                if self.timeout > 0:
                    signal.alarm(0)
            except TimeoutError:
                self.sout(f'Timeout reached for combination {params}.')
                some_timeouts = True
            
            if self.verbose:
                print(f"\t\\--ended evaluation of {str(params)}")
            
            return np.mean(scores) if scores else None

        results = parallel(
            evaluate_combination,
            tqdm(param_combinations, desc="Evaluating combination", total=len(param_combinations)) if self.verbose else param_combinations,
            n_jobs=self.n_jobs
        )
        
        for score, params in zip(results, param_combinations):
            if score is not None and (best_score is None or score < best_score):
                best_score = score
                best_params = params

        self.best_score_ = best_score
        self.best_params_ = dict(zip(self.param_grid.keys(), best_params))
        self.sout(f'Optimization complete. Best score: {self.best_score_}, with parameters: {self.best_params_}.')
        
        if self.refit and self.best_params_:
            self.model.set_params(**self.best_params_)
            self.model.fit(X, y)
            self.best_model_ = self.model

        return self
    
    def predict(self, X):
        if not hasattr(self, 'best_model_'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.best_model_.predict(X)
    
    @property
    def classes_(self):
        return self.best_model_.classes_

    def set_params(self, **parameters):
        self.param_grid = parameters

    def get_params(self, deep=True):
        if hasattr(self, 'best_model_'):
            return self.best_model_.get_params()
        raise ValueError('get_params called before fit')

    def best_model(self):
        if hasattr(self, 'best_model_'):
            return self.best_model_
        raise ValueError('best_model called before fit')
