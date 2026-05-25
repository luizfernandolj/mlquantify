from mlquantify.base import BaseQuantifier, MetaquantifierMixin
import itertools
from joblib import Parallel, delayed
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
from mlquantify.metrics._slq import MAE
from mlquantify.utils._constraints import (
    Interval,
    Options,
    CallableConstraint
)
from mlquantify.utils._validation import validate_data
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils.prevalence import get_prev_from_labels
from mlquantify.model_selection import (
    APP, NPP, UPP
)

class GridSearchQ(MetaquantifierMixin, BaseQuantifier):
    r"""Grid search over quantifier hyperparameters with evaluation protocols.

    Evaluates all combinations in ``param_grid`` using a held-out validation
    split and a sampling protocol (:class:`APP`, :class:`NPP`, or :class:`UPP`).
    Selects the combination with the lowest score on the chosen metric, then
    optionally refits on the full training data.

    Parameters
    ----------
    quantifier : BaseQuantifier
        Quantifier instance whose hyperparameters are searched.
    param_grid : dict
        Mapping of parameter names to lists of values to try.
    protocol : {'app', 'npp', 'upp'}, default='app'
        Evaluation protocol used to generate validation batches.
    samples_sizes : int or list of int, default=100
        Batch size(s) for protocol evaluation.
    n_repetitions : int, default=10
        Number of repetitions per evaluation batch.
    scoring : callable, default=MAE
        Scoring function ``(true_prev, predicted_prev) -> float``.
    refit : bool, default=True
        If ``True``, refit the quantifier on the full data after search.
    val_split : float, default=0.4
        Fraction of data held out for validation.
    n_jobs : int or None, default=1
        Number of parallel evaluation jobs.
    random_seed : int or None, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        Print progress messages.

    Attributes
    ----------
    best_score_ : float
        Lowest score found during the search.
    best_params_ : dict
        Hyperparameter combination that achieved ``best_score_``.
    best_model_ : BaseQuantifier
        Quantifier refitted with ``best_params_`` on the full training data.

    Examples
    --------
    >>> from mlquantify.model_selection import GridSearchQ
    >>> from mlquantify.counting import CC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=300, random_state=42)
    >>> param_grid = {'threshold': [0.3, 0.5, 0.7]}
    >>> gs = GridSearchQ(
    ...     CC(LogisticRegression()),
    ...     param_grid=param_grid,
    ...     protocol='npp',
    ...     n_repetitions=3,
    ... ).fit(X, y)
    >>> gs.best_params_
    {'threshold': 0.5}
    >>> gs.predict(X)
    {0: 0.49, 1: 0.51}
    """
    
    _parameter_constraints = {
        "quantifier": [BaseQuantifier],
        "param_grid": [dict],
        "protocol": [Options({'app', 'npp', 'upp'})],
        "n_samples": [Interval(1, None)],
        "n_repetitions": [Interval(1, None)],
        "scoring": [CallableConstraint()],
        "refit": [bool],
        "val_split": [Interval(0.0, 1.0, inclusive_left=False, inclusive_right=False)],
        "n_jobs": [Interval(-1, None), None],
        "random_seed": [Interval(0, None), None],
        "timeout": [Interval(-1, None)],
        "verbose": [bool]
    }
    
    
    def __init__(self,
                 quantifier,
                 param_grid,
                 protocol="app",
                 samples_sizes=100,
                 n_repetitions=10,
                 scoring=MAE,
                 refit=True,
                 val_split=0.4,
                 n_jobs=1,
                 random_seed=42,
                 verbose=False):
                     
        self.quantifier = quantifier()
        self.param_grid = param_grid
        self.protocol = protocol.lower()
        self.samples_sizes = samples_sizes
        self.n_repetitions = n_repetitions
        self.refit = refit
        self.val_split = val_split
        self.n_jobs = n_jobs
        self.random_seed = random_seed
        self.verbose = verbose
        self.scoring = scoring
        
    
    def sout(self, msg):
        """Prints messages if verbose is True."""
        if self.verbose:
            print(f'[{self.__class__.__name__}]: {msg}')
            
    def __get_protocol(self):
        
        if self.protocol == "app":
            return APP(batch_size=self.samples_sizes,
                       n_prevalences=self.n_repetitions,
                       repeats=self.n_repetitions,
                       random_state=self.random_seed,
                       min_prev=0.0,
                       max_prev=1.0)
        elif self.protocol == "npp":
            return NPP(batch_size=self.samples_sizes,
                       n_samples=self.n_repetitions,
                       repeats=self.n_repetitions,
                       random_state=self.random_seed)
        elif self.protocol == "upp":
            return UPP(batch_size=self.samples_sizes,
                       n_prevalences=self.n_repetitions,
                       repeats=self.n_repetitions,
                       random_state=self.random_seed,
                       min_prev=0.0,
                       max_prev=1.0)
        else:  
            raise ValueError(f'Unknown protocol: {self.protocol}')
        
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """
        Fit quantifiers over grid parameter combinations with evaluation protocol.

        Splits data into training and validation by val_split, and evaluates
        each parameter combination multiple times with protocol-generated batches.

        Parameters
        ----------
        X : array-like
            Feature matrix for training.
        y : array-like
            Target labels for training.

        Returns
        -------
        self : object
            Returns self for chaining.
        """
        X, y = validate_data(self, X, y)
        
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, random_state=self.random_seed)
        param_combinations = list(itertools.product(*self.param_grid.values()))
        params = list(self.param_grid.keys())

        best_score, best_params = None, None
        
        def evaluate_combination(params):
           
            self.sout(f'Evaluating combination: {str(params)}')
            
            errors = []
            
            params = dict(zip(self.param_grid.keys(), params))

            model = deepcopy(self.quantifier)
            model.set_params(**params)
            
            protocol = self.__get_protocol()
            
            model.fit(X_train, y_train)
            
            for idx in protocol.split(X_val, y_val):
                X_batch, y_batch = X_val[idx], y_val[idx]

                y_real = get_prev_from_labels(y_batch)
                y_pred = model.predict(X_batch)

                
                errors.append(self.scoring(y_real, y_pred))

            avg_score = np.mean(errors)
            
            self.sout(f'\\--Finished evaluation: {str(params)} with score: {avg_score}')
            
            return avg_score
                
        
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_combination)(params) for params in param_combinations
        )
        
            
        for score, params in zip(results, param_combinations):
            if score is not None and (best_score is None or score < best_score):
                best_score, best_params = score, params
            
        
        self.best_score = best_score
        self.best_params = dict(zip(self.param_grid.keys(), best_params))
        self.sout(f'Optimization complete. Best score: {self.best_score}, with parameters: {self.best_params}.')

        if self.refit and self.best_params:
            model = deepcopy(self.quantifier)
            model.set_params(**self.best_params)
            model.fit(X, y)
            self.best_model_ = model

        return self
    
    
    
    def predict(self, X):
        """
        Predict using the best found model.

        Parameters
        ----------
        X : array-like
            Data for prediction.

        Returns
        -------
        predictions : array-like
            Prevalence predictions.

        Raises
        ------
        RuntimeError
            If called before fitting.
        """
        if not hasattr(self, 'best_model_'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.best_model_.predict(X)
    
    
    
    def best_params(self):
        """Return the best parameters found during fitting.

        Returns
        -------
        dict
            The best parameters.

        Raises
        ------
        ValueError
            If called before fitting.
        """
        if hasattr(self, 'best_params'):
            return self.best_params
        raise ValueError('best_params called before fit.')
        
        
        
    def best_model(self):
        """Return the best model after fitting.

        Returns
        -------
        Quantifier
            The best fitted model.

        Raises
        ------
        ValueError
            If called before fitting.
        """
        if hasattr(self, 'best_model_'):
            return self.best_model_
        raise ValueError('best_model called before fit.')
