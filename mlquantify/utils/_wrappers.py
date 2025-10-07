from abc import ABC, abstractmethod
from copy import deepcopy
import numpy as np
from joblib import Parallel, delayed

from mlquantify.utils._random import check_random_state
from mlquantify.utils._parallel import resolve_n_jobs


class BaseWrapper(ABC):
    """Abstract base class for quantifier wrappers (OVA, OVO, etc.)."""

    def __init__(self, quantifier, n_jobs=None, random_state=None):
        self.quantifier = quantifier
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y, *args, **kwargs):
        """Fit the wrapper using the provided quantifier and data."""
        self.quantifier.classes = np.unique(y)
        self.quantifier.binary_models = {}
        self._fit_strategy(X, y, *args, **kwargs)
        self.quantifier._original_fit(X, y, *args, **kwargs)

    @abstractmethod
    def _fit_strategy(self, X, y, *args, **kwargs):
        """Define how models are fit depending on the wrapper strategy."""
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs):
        """Define how predictions are made depending on the wrapper strategy."""
        pass
    
    
class OvaWrapper(BaseWrapper):
    """One-vs-All wrapper for multi-class quantifiers."""

    def _fit_strategy(self, X, y, *args, **kwargs):
        classes = self.quantifier.classes
        rng = check_random_state(self.random_state)
        n_jobs = resolve_n_jobs(self.n_jobs)

        def fit_class(_class, seed):
            model = deepcopy(self.quantifier)
            model.random_state = seed
            binary_y = (y == _class).astype(int)
            model._original_fit(X, binary_y, *args, **kwargs)
            self.quantifier.binary_models[_class] = model

        seeds = rng.randint(0, 2**32, size=len(classes))
        Parallel(n_jobs=n_jobs)(
            delayed(fit_class)(_class, seed) for _class, seed in zip(classes, seeds)
        )

    def predict(self, X, *args, **kwargs):
        predictions = {}
        for _class in self.quantifier.classes:
            model = self.quantifier.binary_models[_class]
            predictions[_class] = model._original_predict(X, *args, **kwargs)
        return predictions


class OvoWrapper(BaseWrapper):
    """One-vs-One wrapper for multi-class quantifiers."""

    def _fit_strategy(self, X, y, *args, **kwargs):
        from itertools import combinations

        classes = self.quantifier.classes
        pairs = list(combinations(classes, 2))
        rng = check_random_state(self.random_state)
        n_jobs = resolve_n_jobs(self.n_jobs)

        def fit_pair(pair, seed):
            cls_a, cls_b = pair
            idx = np.where((y == cls_a) | (y == cls_b))[0]
            X_pair = X[idx]
            y_pair = y[idx]
            y_binary = (y_pair == cls_a).astype(int)
            model = deepcopy(self.quantifier)
            model.random_state = seed
            model._original_fit(X_pair, y_binary, *args, **kwargs)
            self.quantifier.binary_models[pair] = model

        seeds = rng.randint(0, 2**32, size=len(pairs))
        Parallel(n_jobs=n_jobs)(
            delayed(fit_pair)(pair, seed) for pair, seed in zip(pairs, seeds)
        )

    def predict(self, X, *args, **kwargs):
        from collections import defaultdict
        from itertools import combinations

        classes = self.quantifier.classes
        pairs = list(combinations(classes, 2))
        n_jobs = resolve_n_jobs(self.n_jobs)

        def predict_pair(pair):
            model = self.quantifier.binary_models[pair]
            return pair, model._original_predict(X, *args, **kwargs)

        results = Parallel(n_jobs=n_jobs)(
            delayed(predict_pair)(pair) for pair in pairs
        )

        # voting mechanism
        votes = defaultdict(lambda: np.zeros(X.shape[0]))
        for (cls_a, cls_b), pred in results:
            for i, p in enumerate(pred):
                votes[cls_a if p == 1 else cls_b][i] += 1

        final_predictions = []
        for i in range(X.shape[0]):
            pred_class = max(classes, key=lambda c: votes[c][i])
            final_predictions.append(pred_class)

        return np.array(final_predictions)
