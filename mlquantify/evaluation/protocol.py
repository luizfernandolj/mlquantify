from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Generator, Union, List, Tuple, Any
from sklearn.base import BaseEstimator
from time import time
from tqdm import tqdm
from itertools import product

from ..methods import METHODS, AGGREGATIVE, NON_AGGREGATIVE
from ..utils.general import *
from ..utils.method import *
from . import MEASURES
from ..base import Quantifier

import mlquantify as mq


def _arg_padronizer()


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

    def __init__(self, n_samples, sample_size, random_state=None):
        if isinstance(n_samples, int):
            n_samples = (1, n_samples)
        if isinstance(sample_size, int):
            sample_size = (sample_size, sample_size)
        if not (isinstance(n_samples, tuple) and len(n_samples) == 2):
            raise ValueError("n_samples must be a tuple of (train, test) sizes.")
        if not (isinstance(sample_size, tuple) and len(sample_size) == 2):
            raise ValueError("sample_size must be a tuple of (train, test) sizes.")
        self.n_samples = n_samples
        self.sample_size = sample_size
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray, *args) -> Generator[np.ndarray, np.ndarray]:

        indices = np.arange(X.shape[0])
        for train_idx, test_idx in self._split_indices_masks(X, y, *args):
            train_index = indices[train_idx]
            test_index = indices[test_idx]
            yield train_index, test_index

    def _dynamic_disjoint_zip(self, X, y, *args):
        seen_tr = set()
        seen_ts = set()
        iter1 = iter(self._iter_train_indices(X, y, *args))
        iter2 = iter(self._iter_test_indices(X, y, *args))

        while True:
            # get next valid value from train
            val1 = None
            while val1 is None:
                try:
                    candidate = next(iter1)
                    if candidate not in seen_ts:
                        val1 = candidate
                        seen_tr.add(val1)
                except StopIteration:
                    return
                
            # get next valid value from test
            val2 = None
            while val2 is None:
                try:
                    candidate = next(iter2)
                    if candidate not in seen_tr:
                        val2 = candidate
                        seen_ts.add(val2)
                except StopIteration:
                    return

            yield val1, val2


    def _split_indices_masks(self, X: np.ndarray, y: np.ndarray, *args) -> Generator[Tuple[np.ndarray, np.ndarray]]:
        for train_idx, test_idx in self._dynamic_disjoint_zip(X, y, *args):
            train_mask = np.zeros(X.shape[0], dtype=bool)
            train_mask[train_idx] = True

            test_mask = np.zeros(X.shape[0], dtype=bool)
            test_mask[test_idx] = True
            
            yield train_mask, test_mask


    @abstractmethod
    def _iter_train_indices(self, X, y):
        pass
    
    @abstractmethod
    def _iter_test_indices(self, X, y):
        pass
    
    @abstractmethod
    def get_n_splits(self) -> int:
        pass


class APP(Protocol):

    def __init__(self, n_prevs, tr_sample_size, ts_sample_size, random_state=None):
        if isinstance(n_prevs, int):
            n_prevs = (1, n_prevs)
        super().__init__(n_samples=n_prevs, sample_size=(tr_sample_size, ts_sample_size), random_state=random_state)

    def _iter_train_indices(self, X: np.ndarray, y: np.ndarray) -> Generator[np.ndarray]:
        n_samples = self.n_samples[0]
        sample_size = self.sample_size[0]
        
        if isinstance(sample_size, int):
            if sample_size > X.shape[0]:
                Warning("Sample size is greater than the number of samples in X, replacement will be applied.")
        
        for _ in range(n_samples):
            prev = generate_artificial_prevalences(n_dim=np.unique(y), n_prev=n_samples, n_iter=1)