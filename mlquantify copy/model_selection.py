from .base import Quantifier
from typing import Union, List
import itertools
from tqdm import tqdm
import signal
from copy import deepcopy
import numpy as np
from sklearn.model_selection import train_test_split
from .utils.general import parallel, get_measure
from .evaluation.protocol import APP, NPP

class GridSearchQ(Quantifier):
    """Hyperparameter optimization for quantification models using grid search.

    GridSearchQ allows hyperparameter tuning for quantification models 
    by minimizing a quantification-oriented loss over a parameter grid. 
    This method evaluates hyperparameter configurations using quantification 
    metrics rather than standard classification metrics, ensuring better 
    approximation of class distributions.

    Parameters
    ----------
    model : Quantifier
        The base quantification model to optimize.

    param_grid : dict
        Dictionary where keys are parameter names (str) and values are 
        lists of parameter settings to try.

    protocol : str, default='app'
        The quantification protocol to use. Supported options are:
        - 'app': Artificial Prevalence Protocol.
        - 'npp': Natural Prevalence Protocol.

    n_prevs : int, default=None
        Number of prevalence points to generate for APP.

    n_repetitions : int, default=1
        Number of repetitions to perform for NPP.

    scoring : Union[List[str], str], default='mae'
        Metric or metrics to evaluate the model's performance. Can be 
        a string (e.g., 'mae') or a list of metrics.

    refit : bool, default=True
        If True, refit the model using the best found hyperparameters 
        on the entire dataset.

    val_split : float, default=0.4
        Proportion of the training data to use for validation. Only 
        applicable if cross-validation is not used.

    n_jobs : int, default=1
        The number of jobs to run in parallel. -1 means using all processors.

    random_seed : int, default=42
        Random seed for reproducibility.

    timeout : int, default=-1
        Maximum time (in seconds) allowed for a single parameter combination.
        A value of -1 disables the timeout.

    verbose : bool, default=False
        If True, print progress messages during grid search.

    Attributes
    ----------
    best_params : dict
        The parameter setting that gave the best results on the validation set.

    best_score : float
        The best score achieved during the grid search.

    results : pandas.DataFrame
        A DataFrame containing details of all evaluations, including parameters, 
        scores, and execution times.

    References
    ----------
    The idea of using grid search for hyperparameter optimization in 
    quantification models was discussed in:
    Moreo, Alejandro; Sebastiani, Fabrizio. "Re-assessing the 'Classify and Count' 
    Quantification Method". In: Advances in Information Retrieval: 
    43rd European Conference on IR Research, ECIR 2021, Virtual Event, 
    March 28–April 1, 2021, Proceedings, Part II. Springer International Publishing, 
    2021, pp. 75–91. [Link](https://link.springer.com/chapter/10.1007/978-3-030-72240-1_6).

    Examples
    --------
    >>> from mlquantify.methods.aggregative import DyS
    >>> from mlquantify.model_selection import GridSearchQ
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> 
    >>> # Loading dataset from sklearn
    >>> features, target = load_breast_cancer(return_X_y=True)
    >>> 
    >>> # Splitting into train and test
    >>> X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)
    >>> 
    >>> model = DyS(RandomForestClassifier())
    >>> 
    >>> # Creating the hyperparameter grid
    >>> param_grid = {
    >>>     'learner__n_estimators': [100, 500, 1000],
    >>>     'learner__criterion': ["gini", "entropy"],
    >>>     'measure': ["topsoe", "hellinger"]
    >>> }
    >>> 
    >>> gs = GridSearchQ(
    ...                 model=model,
    ...                 param_grid=param_grid,
    ...                 protocol='app', # Default
    ...                 n_prevs=100,    # Default
    ...                 scoring='nae',
    ...                 refit=True,     # Default
    ...                 val_split=0.3,
    ...                 n_jobs=-1,
    ...                 verbose=True)
    >>> 
    >>> gs.fit(X_train, y_train)
    [GridSearchQ]: Optimization complete. Best score: 0.0060630241297973545, with parameters: {'learner__n_estimators': 500, 'learner__criterion': 'entropy', 'measure': 'topsoe'}.
    >>> predictions = gs.predict(X_test)
    >>> predictions
    {0: 0.4182508973311534, 1: 0.5817491026688466}
    """


    def __init__(self,
                 model: Quantifier,
                 param_grid: dict,
                 protocol: str = 'app',
                 n_prevs: int = 100,
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

        Parameters
        ----------
        model : Quantifier
            The quantification model.

        sample_size : int
            The sample size for batch processing.

        Returns
        -------
        object
            Instance of APP or NPP protocol, depending on the configured protocol.
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

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : GridSearchQ
            Returns the fitted instance of GridSearchQ.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_split, random_state=self.random_seed)
        param_combinations = list(itertools.product(*self.param_grid.values()))
        best_score, best_params = None, None

        if self.timeout > 0:
            signal.signal(signal.SIGALRM, self._timeout_handler)

        def evaluate_combination(params):
            """Evaluate a single combination of hyperparameters.

            Parameters
            ----------
            params : tuple
                A tuple of hyperparameter values.

            Returns
            -------
            float or None
                The evaluation score, or None if a timeout occurred.
            """
            if self.verbose:
                print(f"\tEvaluating combination: {str(params)}")

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
                    print(f"\t\\--Finished evaluation: {str(params)}")

                return np.mean(scores) if scores else None
            except TimeoutError:
                self.sout(f'Timeout reached for combination: {params}.')
                return None

        results = parallel(
            evaluate_combination,
            tqdm(param_combinations, desc="Evaluating combinations", total=len(param_combinations)) if self.verbose else param_combinations,
            n_jobs=self.n_jobs
        )

        for score, params in zip(results, param_combinations):
            if score is not None and (best_score is None or score < best_score):
                best_score, best_params = score, params

        self.best_score = best_score
        self.best_params = dict(zip(self.param_grid.keys(), best_params))
        self.sout(f'Optimization complete. Best score: {self.best_score}, with parameters: {self.best_params}.')

        if self.refit and self.best_params:
            self.model.set_params(**self.best_params)
            self.model.fit(X, y)
            self.best_model_ = self.model

        return self

    def predict(self, X):
        """Make predictions using the best found model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.

        Returns
        -------
        array-like
            Predictions for the input data.
        
        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if not hasattr(self, 'best_model_'):
            raise RuntimeError("The model has not been fitted yet.")
        return self.best_model_.predict(X)

    @property
    def classes_(self):
        """Get the classes of the best model.

        Returns
        -------
        array-like
            The classes learned by the best model.
        """
        return self.best_model_.classes_

    def set_params(self, **parameters):
        """Set the hyperparameters for grid search.

        Parameters
        ----------
        parameters : dict
            Dictionary of hyperparameters to set.
        """
        self.param_grid = parameters

    def get_params(self, deep=True):
        """Get the parameters of the best model.

        Parameters
        ----------
        deep : bool, optional, default=True
            If True, will return the parameters for this estimator and 
            contained subobjects.

        Returns
        -------
        dict
            Parameters of the best model.

        Raises
        ------
        ValueError
            If called before the model has been fitted.
        """
        if hasattr(self, 'best_model_'):
            return self.best_model_.get_params()
        raise ValueError('get_params called before fit.')

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

    def _timeout_handler(self, signum, frame):
        """Handle timeouts during evaluation.

        Parameters
        ----------
        signum : int
            Signal number.

        frame : object
            Current stack frame.

        Raises
        ------
        TimeoutError
            Raised when the timeout is reached.
        """
        raise TimeoutError
