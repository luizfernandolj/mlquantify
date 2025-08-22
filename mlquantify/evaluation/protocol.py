from abc import ABC, abstractmethod
import numpy as np
from typing import Generator, Tuple
from tqdm import tqdm

from ..utils.general import *

class Protocol(ABC):
    """Base class for evaluation protocols.
    
    Parameters
    ----------
        
    Attributes
    ----------

    Raises
    ------

    Notes
    -----

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

        indices = np.arange(X.shape[0])
        for idx in self._split_indices_masks(X, y):
            indexes = indices[idx]
            yield indexes

    def _split_indices_masks(self, X: np.ndarray, y: np.ndarray) -> Generator[Tuple[np.ndarray, np.ndarray]]:
        for idx in self._iter_indices(X, y):

            mask = np.zeros(X.shape[0], dtype=bool)
            mask[idx] = True 

            yield mask

    @abstractmethod
    def _iter_indices(self, X, y):
        pass
    
    def get_n_combinations(self) -> int:
        return self.n_combinations


class APP(Protocol):

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

    def __init__(self, batch_size, random_state=None):
        super().__init__(batch_size=batch_size, random_state=random_state)
        
    def _iter_indices(self, X: np.ndarray, y: np.ndarray) -> Generator[np.ndarray]:
        
        for batch_size in self.batch_size:
            yield np.random.choice(X.shape[0], batch_size, replace=True)
            

class UPP(Protocol):
    
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
        