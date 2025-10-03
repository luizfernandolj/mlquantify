from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
from copy import deepcopy
import numpy as np
import joblib
from functools import wraps

from mlquantify.utils.general import parallel
from mlquantify.utils._tags import (
    Tags,
    TargetInputTags,
    get_tags
)

from abc import ABC, abstractmethod

class BaseWrapper(ABC):

    def fit(self, X, y, *args, **kwargs):
        self.quantifier.classes = np.unique(y)
        self.quantifier.binary_models = {}
        self._fit_strategy(X, y, *args, **kwargs)
        self.quantifier._original_fit(X, y, *args, **kwargs)

    @abstractmethod
    def _fit_strategy(self, X, y, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs):
        pass


class OvaWrapper(BaseWrapper):
    def _fit_strategy(self, X, y, *args, **kwargs):
        classes = self.quantifier.classes

        def fit_class(_class):
            model = deepcopy(self.quantifier)
            binary_y = (y == _class).astype(int)
            model._original_fit(X, binary_y, *args, **kwargs)
            self.quantifier.binary_models[_class] = model

        parallel(fit_class, classes, n_jobs=-1)

    def predict(self, X, *args, **kwargs):
        predictions = {}
        for _class in self.quantifier.classes:
            predictions[_class] = self.quantifier.binary_models[_class]._original_predict(X, *args, **kwargs)
        return predictions


class OvoWrapper(BaseWrapper):
    def _fit_strategy(self, X, y, *args, **kwargs):
        from itertools import combinations

        classes = self.quantifier.classes
        pairs = list(combinations(classes, 2))

        def fit_pair(pair):
            cls_a, cls_b = pair
            idx = np.where((y == cls_a) | (y == cls_b))[0]
            X_pair = X[idx]
            y_pair = y[idx]
            y_binary = (y_pair == cls_a).astype(int)
            model = deepcopy(self.quantifier)
            model._original_fit(X_pair, y_binary, *args, **kwargs)
            self.quantifier.binary_models[pair] = model

        parallel(fit_pair, pairs, n_jobs=-1)

    def predict(self, X, *args, **kwargs):
        from collections import defaultdict
        from itertools import combinations

        classes = self.quantifier.classes
        pairs = list(combinations(classes, 2))
        votes = defaultdict(lambda: np.zeros(X.shape[0]))

        def predict_pair(pair):
            return pair, self.quantifier.binary_models[pair]._original_predict(X, *args, **kwargs)

        results = parallel(predict_pair, pairs, n_jobs=-1)

        for (cls_a, cls_b), pred in results:
            for i, p in enumerate(pred):
                if p == 1:
                    votes[cls_a][i] += 1
                else:
                    votes[cls_b][i] += 1

        final_predictions = []
        for i in range(X.shape[0]):
            pred_class = max(classes, key=lambda c: votes[c][i])
            final_predictions.append(pred_class)

        return np.array(final_predictions)


class BaseQuantifier(ABC, BaseEstimator):
    """Base class for all quantifiers, it defines the basic structure of a quantifier.
    
    Warning: Inheriting from this class does not provide dynamic use of multiclass or binary methods, it is necessary to implement the logic in the quantifier itself. If you want to use this feature, inherit from AggregativeQuantifier or NonAggregativeQuantifier.
    
    Inheriting from this class, it provides the following implementations:
    
    - properties for classes, n_class, is_multiclass and binary_data.
    - save_quantifier method to save the quantifier
    
    Read more in the :ref:`User Guide <creating_your_own_quantifier>`.
    
    
    Notes
    -----
    It's recommended to inherit from AggregativeQuantifier or NonAggregativeQuantifier, as they provide more functionality and flexibility for quantifiers.
    """    
    
    @abstractmethod
    def fit(self, X, y) -> object: ...
    
    @abstractmethod
    def predict(self, X) -> dict: ...
    
    def save_quantifier(self, path: str=None) -> None:
        if not path:
            path = f"{self.__class__.__name__}.joblib"
        joblib.dump(self, path)
        
    def __mlquantify_tags__(self):
        return Tags(
            estimator=None,
            estimation_type=None,
            estimator_function=None,
            estimator_type=None,
            aggregation_type=None,
            target_input_tags=TargetInputTags()
        )


def handle_domain_fit(func):
    @wraps(func)
    def wrapper(self, X, y, *args, **kwargs):
        if isinstance(self, BinaryQMixin):
            domain_mode = getattr(self, 'domain_mode', 'ova').lower()
            if domain_mode == 'ova':
                wrapper_obj = OvaWrapper(self)
            elif domain_mode == 'ovo':
                wrapper_obj = OvoWrapper(self)
            else:
                raise ValueError(f"Unsupported domain_mode: {domain_mode}")
            return wrapper_obj.fit(X, y, *args, **kwargs)
        else:
            return func(self, X, y, *args, **kwargs)
    return wrapper


def handle_domain_predict(func):
    @wraps(func)
    def wrapper(self, X, *args, **kwargs):
        if isinstance(self, BinaryQMixin):
            strategy = getattr(self, 'strategy').lower()
            if strategy == 'ova':
                wrapper_obj = OvaWrapper(self)
            elif strategy == 'ovo':   
                wrapper_obj = OvoWrapper(self)
            else:
                raise ValueError(f"Unsupported domain_mode: {strategy}")
            return wrapper_obj.predict(X, *args, **kwargs)
        else:
            return func(self, X, *args, **kwargs)
    return wrapper

class BinaryQMixin:

    _strategies = ["ova", "ovo"]

    @abstractmethod
    def __init__(self, strategy="ova", *args, **kwargs):
        assert strategy in self._strategies, f"Invalid strategy: {strategy}. Choose from {self._strategies}."
        print("Initializing BinaryQMixin with strategy:", strategy)
        super().__init__(*args, **kwargs)
        self.strategy = strategy

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.target_input_tags = TargetInputTags(multi_class=False)
        return tags

    @handle_domain_fit
    def fit(self, X, y, *args, **kwargs):
        ...

    @handle_domain_predict
    def predict(self, X, *args, **kwargs):
        ...

class SoftLearnerQMixin:
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator = True
        tags.estimator_function = "predict_proba"
        tags.estimator_type = "soft"
        return tags

class CrispLearnerQMixin:

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator = True
        tags.estimator_function = "predict"
        tags.estimator_type= "crisp"
        return tags
    

class RegressorQMixin:
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__() 
        tags.estimator = True
        tags.estimator_function = "predict"
        tags.estimator_type= "regression"
        return tags

class DistributionMixin:

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator = True
        tags.estimation_type = "distribution"
        return tags

class MaximumLikelihoodMixin:

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator = True
        tags.estimation_type = "likelihood"
        return tags

class ThresholdAdjustmentMixin:

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator = True
        tags.estimation_type = "adjusting"
        return tags




def uses_soft_predictions(quantifier):
    return get_tags(quantifier).estimator_type == "soft"

def uses_crisp_predictions(quantifier):
    return get_tags(quantifier).estimator_type == "crisp"

def is_aggregative(quantifier):
    return get_tags(quantifier).learner == True