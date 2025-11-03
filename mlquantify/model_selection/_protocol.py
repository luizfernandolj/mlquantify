import numpy as np

from mlquantify.base import BaseQuantifier, ProtocolMixin
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._sampling import (
    get_indexes_with_prevalence, 
    simplex_grid_sampling,
    simplex_uniform_kraemer,
    simplex_uniform_sampling,
)
from mlquantify.utils._validation import validate_data
from abc import ABC, abstractmethod
from logging import warning
import numpy as np

    
class BaseProtocol(ProtocolMixin, BaseQuantifier):
    """Base class for evaluation protocols.
    
    Parameters
    ----------
    batch_size : int or list of int
        The size of the batches to be used in the evaluation.
    random_state : int, optional
        The random seed for reproducibility.

    Attributes
    ----------
    n_combinations : int

    Raises
    ------
    ValueError
        If the batch size is not a positive integer or list of positive integers.

    Notes
    -----
    This class serves as a base class for different evaluation protocols, each with its own strategy for splitting the data into batches.
    
    Examples
    --------
    >>> class MyCustomProtocol(Protocol):
    ...     def _iter_indices(self, X: np.ndarray, y: np.ndarray):
    ...         for batch_size in self.batch_size:
    ...             yield np.random.choice(X.shape[0], batch_size, replace=True)
    ...
    >>> protocol = MyCustomProtocol(batch_size=100, random_state=42)
    >>> for idx in protocol.split(X, y):
    ...     # Train and evaluate model
    ...     pass

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
        """
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
    
    def get_n_combinations(self) -> int:
        """
        Get the number of combinations for the current protocol.
        """
        return self.n_combinations



# ===========================================
# Protocol Implementations
# ===========================================


class APP(BaseProtocol):
    """Artificial Prevalence Protocol (APP) for evaluation.
    This protocol generates artificial prevalence distributions for the evaluation in an exhaustive manner, testing all possible combinations of prevalences.
    
    Parameters
    ----------
    batch_size : int or list of int
        The size of the batches to be used in the evaluation.
    n_prevalences : int
        The number of artificial prevalences to generate.
    repeats : int, optional
        The number of times to repeat the evaluation with different random seeds.
    random_state : int, optional
        The random seed for reproducibility.
        
    Attributes
    ----------
    n_prevalences : int
        The number of artificial prevalences to generate.
    repeats : int
        The number of times to repeat the evaluation with different random seeds.
    random_state : int
        The random seed for reproducibility.
        
    Notes
    -----
    It is important to note that in case of multiclass problems, the time complexity of this protocol can be significantly higher due to the increased number of combinations to evaluate.

    Examples
    --------
    >>> protocol = APP(batch_size=[100, 200], n_prevalences=5, repeats=3, random_state=42)
    >>> for train_idx, test_idx in protocol.split(X, y):
    ...     # Train and evaluate model
    ...     pass

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
        
        for batch_size in self.batch_size:
            prevalences = simplex_grid_sampling(n_dim=n_dim,
                                              n_prev=self.n_prevalences,
                                              n_iter=self.repeats,
                                              min_val=self.min_prev,
                                              max_val=self.max_prev)
            for prev in prevalences:
                indexes = get_indexes_with_prevalence(y, prev, batch_size)
                yield indexes
                

            

class NPP(BaseProtocol):
    """No Prevalence Protocol (NPP) for evaluation.
    This protocol just samples the data without any consideration for prevalence, with all instances having equal probability of being selected.

    Parameters
    ----------
    batch_size : int or list of int
        The size of the batches to be used in the evaluation.
    random_state : int, optional
        The random seed for reproducibility.

    Attributes
    ----------
    n_prevalences : int
        The number of artificial prevalences to generate.
    repeats : int
        The number of times to repeat the evaluation with different random seeds.
    random_state : int
        The random seed for reproducibility.

    Examples
    --------
    >>> protocol = NPP(batch_size=100, random_state=42)
    >>> for train_idx, test_idx in protocol.split(X, y):
    ...     # Train and evaluate model
    ...     pass
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
        
        for _ in range(self.n_samples):
            for batch_size in self.batch_size:
                idx = np.random.choice(X.shape[0], batch_size, replace=True)
                for _ in range(self.repeats):
                    yield idx
            

class UPP(BaseProtocol):
    """Uniform Prevalence Protocol (UPP) for evaluation.
    An extension of the APP that generates artificial prevalence distributions uniformly across all classes utilizing the kraemer sampling method.

    Parameters
    ----------
    batch_size : int or list of int
        The size of the batches to be used in the evaluation.
    n_prevalences : int
        The number of artificial prevalences to generate.
    repeats : int
        The number of times to repeat the evaluation with different random seeds.
    random_state : int, optional
        The random seed for reproducibility.

    Attributes
    ----------
    n_prevalences : int
        The number of artificial prevalences to generate.
    repeats : int
        The number of times to repeat the evaluation with different random seeds.
    random_state : int
        The random seed for reproducibility.

    Examples
    --------
    >>> protocol = UPP(batch_size=100, n_prevalences=5, repeats=3, random_state=42)
    >>> for train_idx, test_idx in protocol.split(X, y):
    ...     # Train and evaluate model
    ...     pass
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
        
        for batch_size in self.batch_size:
            if self.algorithm == 'kraemer':
                prevalences = simplex_uniform_kraemer(n_dim=n_dim,
                                           n_prev=self.n_prevalences,
                                           n_iter=self.repeats,
                                           min_val=self.min_prev,
                                           max_val=self.max_prev)
            elif self.algorithm == 'uniform':
                prevalences = simplex_uniform_sampling(n_dim=n_dim,
                                              n_prev=self.n_prevalences,
                                              n_iter=self.repeats,
                                              min_val=self.min_prev,
                                              max_val=self.max_prev)

            for prev in prevalences:
                indexes = get_indexes_with_prevalence(y, prev, batch_size)
                yield indexes


class PPP(BaseProtocol):
    """ Personalized Prevalence Protocol (PPP) for evaluation.
    This protocol generates artificial prevalence distributions personalized for each class.

    Parameters
    ----------
    batch_size : int or list of int
        The size of the batches to be used in the evaluation.
    prevalences : list of float
        The list of artificial prevalences to generate for each class.
    repeats : int
        The number of times to repeat the evaluation with different random seeds.
    random_state : int, optional
        The random seed for reproducibility.

    Attributes
    ----------
    prevalences : list of float
        The list of artificial prevalences to generate for each class.
    repeats : int
        The number of times to repeat the evaluation with different random seeds.
    random_state : int
        The random seed for reproducibility.

    Examples
    --------
    >>> protocol = PPP(batch_size=100, prevalences=[0.1, 0.9], repeats=3, random_state=42)
    >>> for train_idx, test_idx in protocol.split(X, y):
    ...     # Train and evaluate model
    ...     pass
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
        
        for batch_size in self.batch_size:    
            for prev in self.prevalences:
                if isinstance(prev, float):
                    prev = [1-prev, prev]
                
                indexes = get_indexes_with_prevalence(y, prev, batch_size)
                yield indexes
        