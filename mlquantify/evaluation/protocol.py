from abc import ABC, abstractmethod
from logging import warning
import numpy as np
from typing import Generator, Tuple
from tqdm import tqdm

from ..utils.general import *

class Protocol(ABC):
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
    ...     def _iter_indices(self, X: np.ndarray, y: np.ndarray) -> Generator[np.ndarray]:
    ...         for batch_size in self.batch_size:
    ...             yield np.random.choice(X.shape[0], batch_size, replace=True)
    ...
    >>> protocol = MyCustomProtocol(batch_size=100, random_state=42)
    >>> for train_idx, test_idx in protocol.split(X, y):
    ...     # Train and evaluate model
    ...     pass

    """

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
                 

    def split(self, X: np.ndarray, y: np.ndarray) -> Generator[np.ndarray, np.ndarray]:
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


class APP(Protocol):
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

    def __init__(self, batch_size, n_prevalences, repeats=1, random_state=None):
        super().__init__(batch_size=batch_size, 
                            random_state=random_state,
                            n_prevalences=n_prevalences, 
                            repeats=repeats)

    def _iter_indices(self, X: np.ndarray, y: np.ndarray) -> Generator[np.ndarray]:
        
        n_dim = len(np.unique(y))
        
        for batch_size in self.batch_size:
            prevalences = generate_artificial_prevalences(n_dim=n_dim,
                                                           n_prev=self.n_prevalences,
                                                           n_iter=self.repeats)
            for prev in prevalences:
                indexes = get_indexes_with_prevalence(y, prev, batch_size)
                yield indexes
                

            

class NPP(Protocol):
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

    def _iter_indices(self, X: np.ndarray, y: np.ndarray) -> Generator[np.ndarray]:
        
        for batch_size in self.batch_size:
            yield np.random.choice(X.shape[0], batch_size, replace=True)
            

class UPP(Protocol):
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

    def __init__(self, batch_size, n_prevalences, repeats=1, random_state=None):
        super().__init__(batch_size=batch_size, 
                            random_state=random_state,
                            n_prevalences=n_prevalences, 
                            repeats=repeats)

    def _iter_indices(self, X: np.ndarray, y: np.ndarray) -> Generator[np.ndarray]:
        
        n_dim = len(np.unique(y))
        
        for batch_size in self.batch_size:
            
            prevalences = kraemer_sampling(n_dim=n_dim,
                                           n_prev=self.n_prevalences,
                                           n_iter=self.repeats)
            
            for prev in prevalences:
                indexes = get_indexes_with_prevalence(y, prev, batch_size)
                yield indexes


class PPP(Protocol):
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
    
    def __init__(self, batch_size, prevalences, repeats=1, random_state=None):
        super().__init__(batch_size=batch_size, 
                        random_state=random_state,
                        prevalences=prevalences, 
                        repeats=repeats)
    
    def _iter_indices(self, X: np.ndarray, y: np.ndarray) -> Generator[np.ndarray]:
        
        for batch_size in self.batch_size:    
            for prev in self.prevalences:
                if isinstance(prev, float):
                    prev = [1-prev, prev]
                
                indexes = get_indexes_with_prevalence(y, prev, batch_size)
                yield indexes
        