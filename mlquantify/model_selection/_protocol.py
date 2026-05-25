import numpy as np

from mlquantify.base import BaseQuantifier, ProtocolMixin
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._sampling import (
    get_indexes_with_prevalence, 
    simplex_grid_sampling,
    simplex_uniform_kraemer,
    simplex_uniform_sampling,
)
from mlquantify.utils._random import check_random_state
from mlquantify.utils._validation import validate_data
from abc import ABC, abstractmethod
from logging import warning
import numpy as np

    
class BaseProtocol(ProtocolMixin, BaseQuantifier):
    r"""Abstract base class for evaluation protocols.

    Provides the :meth:`split` interface that yields sample indices for
    evaluating a quantifier across varying prevalence conditions. Subclasses
    implement :meth:`_iter_indices` to define the specific sampling strategy.

    Parameters
    ----------
    batch_size : int or list of int
        Size(s) of the evaluation batches.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    n_combinations : int
        Total number of batches this protocol will generate.

    Examples
    --------
    >>> from mlquantify.model_selection._protocol import BaseProtocol
    >>> import numpy as np
    >>> class MyProtocol(BaseProtocol):
    ...     def _iter_indices(self, X, y):
    ...         rng = np.random.default_rng(self.random_state)
    ...         for bs in self.batch_size:
    ...             yield rng.choice(len(X), bs, replace=True)
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = MyProtocol(batch_size=50, random_state=0)
    >>> idx = next(proto.split(X, y))
    >>> len(idx)
    50
    """
    
    _parameter_constraints = {
        "batch_size": [Interval(left=1, right=None, discrete=True)],
        "random_state": [Interval(left=0, right=None, discrete=True)]
    }

    def __init__(self, batch_size, random_state=None, **kwargs):
        if isinstance(batch_size, int):
            self.n_combinations = 1
        else:
            self.n_combinations = len(batch_size)

        self.batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.random_state = random_state

        for name, value in kwargs.items():
            setattr(self, name, value)
            if isinstance(value, list):
                self.n_combinations *= len(value)
            elif isinstance(value, (int, float)):
                self.n_combinations *= value
            else:
                raise ValueError(f"Invalid argument {name}={value}: must be int/float or list of int/float.")
                 

    def split(self, X: np.ndarray, y: np.ndarray):
        r"""
        Split the data into samples for evaluation.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The target labels.

        Yields
        ------
        Generator[np.ndarray, np.ndarray]
            A generator that yields the indices for each split.
        """
        X, y = validate_data(self, X, y)
        for idx in self._iter_indices(X, y):
            if len(idx) > len(X):
                warning(f"Batch size {len(idx)} exceeds dataset size {len(X)}. Replacement sampling will be used.")
            yield idx


    @abstractmethod
    def _iter_indices(self, X, y):
        """Abstract method to be implemented by subclasses to yield indices for each batch."""
        pass
    
    def get_n_combinations(self):
        """
        Get the number of combinations for the current protocol.
        """
        return self.n_combinations



# ===========================================
# Protocol Implementations
# ===========================================


class APP(BaseProtocol):
    r"""Artificial Prevalence Protocol (APP).

    Generates evaluation batches with artificially imposed prevalences sampled
    on a regular grid over the probability simplex within ``[min_prev, max_prev]``.
    Covers all combinations of prevalence levels for comprehensive evaluation.

    Parameters
    ----------
    batch_size : int or list of int
        Size(s) of the evaluation batches.
    n_prevalences : int
        Number of prevalence grid points per class dimension.
    repeats : int, default=1
        Number of repetitions for each prevalence combination.
    random_state : int or None, default=None
        Random seed for reproducibility.
    min_prev : float, default=0.0
        Minimum class prevalence.
    max_prev : float, default=1.0
        Maximum class prevalence.

    Attributes
    ----------
    n_combinations : int
        Total number of batches generated.

    Notes
    -----
    For multiclass problems the grid grows combinatorially; prefer :class:`UPP`
    for large class counts.

    Examples
    --------
    >>> from mlquantify.model_selection import APP
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = APP(batch_size=50, n_prevalences=5, random_state=0)
    >>> batches = list(proto.split(X, y))
    >>> len(batches)
    6

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2008). Quantifying Counts and Costs via Classification.
               *Data Mining and Knowledge Discovery*, 17(2), 164–206.
        .. [2] Sebastiani, F., et al. (2020). A Critical Reassessment of the
               Evaluation of Machine Learning Approaches for Quantification.
               *ArXiv preprint*.
    """
    
    _parameter_constraints = {
        "n_prevalences": [Interval(left=1, right=None, discrete=True)],
        "repeats": [Interval(left=1, right=None, discrete=True)],
        "min_prev": [Interval(left=0.0, right=1.0)],
        "max_prev": [Interval(left=0.0, right=1.0)]
    }

    def __init__(self, batch_size, n_prevalences, repeats=1, random_state=None, min_prev=0.0, max_prev=1.0):
        super().__init__(batch_size=batch_size, 
                            random_state=random_state,
                            n_prevalences=n_prevalences, 
                            repeats=repeats)
        self.min_prev = min_prev
        self.max_prev = max_prev

    def _iter_indices(self, X: np.ndarray, y: np.ndarray):
        
        n_dim = len(np.unique(y))

        rng = check_random_state(self.random_state)
        
        for batch_size in self.batch_size:
            prevalences = simplex_grid_sampling(n_dim=n_dim,
                                              n_prev=self.n_prevalences,
                                              n_iter=self.repeats,
                                              min_val=self.min_prev,
                                              max_val=self.max_prev)
            for prev in prevalences:
                indexes = get_indexes_with_prevalence(y, prev, batch_size, random_state=rng)
                yield indexes

            

class NPP(BaseProtocol):
    r"""Natural Prevalence Protocol (NPP).

    Samples evaluation batches uniformly at random from the dataset, preserving
    the natural class distribution without imposing any prevalence constraints.

    Parameters
    ----------
    batch_size : int or list of int
        Size(s) of the evaluation batches.
    n_samples : int, default=1
        Number of distinct batch samples per batch size.
    repeats : int, default=1
        Number of repetitions for each sample.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    n_combinations : int
        Total number of batches generated.

    Examples
    --------
    >>> from mlquantify.model_selection import NPP
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = NPP(batch_size=50, n_samples=3, random_state=0)
    >>> batches = list(proto.split(X, y))
    >>> len(batches)
    3
    """
    
    _parameter_constraints = {
        "repeats": [Interval(left=1, right=None, discrete=True)]
    }
    
    def __init__(self, batch_size, n_samples=1, repeats=1, random_state=None):
        super().__init__(batch_size=batch_size, 
                        random_state=random_state)
        self.n_samples = n_samples
        self.repeats = repeats

    def _iter_indices(self, X: np.ndarray, y: np.ndarray):
        rng = check_random_state(self.random_state)
        for _ in range(self.n_samples):
            for batch_size in self.batch_size:
                idx = rng.choice(X.shape[0], batch_size, replace=True)
                for _ in range(self.repeats):
                    yield idx
            

class UPP(BaseProtocol):
    r"""Uniform Prevalence Protocol (UPP).

    Similar to :class:`APP`, but samples prevalences uniformly over the
    probability simplex rather than on a regular grid, avoiding bias towards
    the simplex corners. Supports Kraemer or uniform simplex sampling.

    Parameters
    ----------
    batch_size : int or list of int
        Batch size(s) for evaluation.
    n_prevalences : int
        Number of prevalence points to sample.
    repeats : int, default=1
        Number of repetitions for each prevalence point.
    random_state : int or None, default=None
        Random seed for reproducibility.
    min_prev : float, default=0.0
        Minimum class prevalence.
    max_prev : float, default=1.0
        Maximum class prevalence.
    algorithm : {'kraemer', 'uniform'}, default='kraemer'
        Simplex sampling algorithm. ``'kraemer'`` uses the Kraemer method;
        ``'uniform'`` uses uniform Dirichlet sampling.

    Attributes
    ----------
    n_combinations : int
        Total number of batches generated.

    Examples
    --------
    >>> from mlquantify.model_selection import UPP
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = UPP(batch_size=50, n_prevalences=5, random_state=0)
    >>> batches = list(proto.split(X, y))
    >>> len(batches)
    5
    """
    
    _parameter_constraints = {
        "n_prevalences": [Interval(left=1, right=None, discrete=True)],
        "repeats": [Interval(left=1, right=None, discrete=True)],
        "min_prev": [Interval(left=0.0, right=1.0)],
        "max_prev": [Interval(left=0.0, right=1.0)],
        "algorithm": [Options(['kraemer', 'uniform'])]
    }

    def __init__(self, 
                 batch_size, 
                 n_prevalences, 
                 repeats=1, 
                 random_state=None, 
                 min_prev=0.0, 
                 max_prev=1.0,
                 algorithm='kraemer'):
        super().__init__(batch_size=batch_size, 
                            random_state=random_state,
                            n_prevalences=n_prevalences, 
                            repeats=repeats)
        self.min_prev = min_prev
        self.max_prev = max_prev
        self.algorithm = algorithm

    def _iter_indices(self, X: np.ndarray, y: np.ndarray):
        
        n_dim = len(np.unique(y))

        rng = check_random_state(self.random_state)
        
        for batch_size in self.batch_size:
            if self.algorithm == 'kraemer':
                prevalences = simplex_uniform_kraemer(n_dim=n_dim,
                                           n_prev=self.n_prevalences,
                                           n_iter=self.repeats,
                                           min_val=self.min_prev,
                                           max_val=self.max_prev,
                                           random_state=rng)
            elif self.algorithm == 'uniform':
                prevalences = simplex_uniform_sampling(n_dim=n_dim,
                                              n_prev=self.n_prevalences,
                                              n_iter=self.repeats,
                                              min_val=self.min_prev,
                                              max_val=self.max_prev,
                                              random_state=rng)
            for prev in prevalences:
                indexes = get_indexes_with_prevalence(y, prev, batch_size, random_state=rng)
                yield indexes


class PPP(BaseProtocol):
    r"""Personalized Prevalence Protocol (PPP).

    Generates evaluation batches with explicitly specified class prevalences,
    enabling controlled evaluation at exact target operating points.

    Parameters
    ----------
    batch_size : int or list of int
        Batch sizes to generate.
    prevalences : list of float or array-like
        Target class prevalences. A single float is interpreted as the positive
        class prevalence in binary problems (negative = 1 - float).
    repeats : int, default=1
        Number of repetitions per prevalence point.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    n_combinations : int
        Total number of batches generated.

    Examples
    --------
    >>> from mlquantify.model_selection import PPP
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = PPP(batch_size=50, prevalences=[[0.2, 0.8], [0.5, 0.5]], random_state=0)
    >>> batches = list(proto.split(X, y))
    >>> len(batches)
    2
    """
    
    _parameter_constraints = {
        "repeats": [Interval(left=1, right=None, discrete=True)],
        "prevalences": ["array-like"]
    }
    
    def __init__(self, batch_size, prevalences, repeats=1, random_state=None):
        super().__init__(batch_size=batch_size, 
                        random_state=random_state,
                        prevalences=prevalences, 
                        repeats=repeats)
    
    def _iter_indices(self, X: np.ndarray, y: np.ndarray):
        rng = check_random_state(self.random_state)
        for batch_size in self.batch_size:    
            for prev in self.prevalences:
                if isinstance(prev, float):
                    prev = [1-prev, prev]
                
                indexes = get_indexes_with_prevalence(y, prev, batch_size, random_state=rng)
                yield indexes
        