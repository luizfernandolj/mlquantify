from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
from copy import deepcopy
import numpy as np
import joblib
from functools import wraps

import mlquantify as mq
from .utils.general import parallel, normalize_prevalence


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


def handle_domain_fit(func):
    @wraps(func)
    def wrapper(self, X, y, *args, **kwargs):
        if getattr(self, 'is_binary_method', False):
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
        if getattr(self, 'is_binary_method', False):
            domain_mode = getattr(self, 'domain_mode', 'ova').lower()
            if domain_mode == 'ova':
                wrapper_obj = OvaWrapper(self)
            elif domain_mode == 'ovo':
                wrapper_obj = OvoWrapper(self)
            else:
                raise ValueError(f"Unsupported domain_mode: {domain_mode}")
            return wrapper_obj.predict(X, *args, **kwargs)
        else:
            return func(self, X, *args, **kwargs)
    return wrapper


class Quantifier(ABC, BaseEstimator):
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
    
    is_binary_method = False
    domain_mode = 'ova'       
    
    @handle_domain_fit
    @abstractmethod
    def fit(self, X, y) -> object: ...
    
    @handle_domain_predict
    @abstractmethod
    def predict(self, X) -> dict: ...
    
    def save_quantifier(self, path: str=None) -> None:
        if not path:
            path = f"{self.__class__.__name__}.joblib"
        joblib.dump(self, path)
        


class AggregativeQuantifier(Quantifier, ABC):
    """A base class for aggregative quantifiers.
    
    This class provides the basic structure for aggregative quantifiers, which are quantifiers that aggregates a classifier or learner inside to generate predictions.
    
    Inheriting from this class, it provides dynamic prediction for multiclass and binary data, making one-vs-all strategy for multiclass data with binary quantifiers.
    
    Read more in the :ref:`User Guide <creating_your_own_quantifier>`.
    
    
    Notes
    -----
    All quantifiers should specify at least the learner attribute. Wich should inherit from BaseEstimator of scikit-learn.
    
    All quantifiers can return a dictionary with class:prevalence, a list or a numpy array.

    
    Examples
    --------
    Example 1: Multiclass Quantifier
    >>> from mlquantify.base import AggregativeQuantifier
    >>> from mlquantify.utils.general import get_real_prev
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> import numpy as np
    >>> class MyQuantifier(AggregativeQuantifier):
    ...     def __init__(self, learner, *, param):
    ...         self.learner = learner
    ...         self.param = param
    ...     def _fit_method(self, X, y):
    ...         self.learner.fit(X, y)
    ...         return self
    ...     def _predict_method(self, X):
    ...         predicted_labels = self.learner.predict(X)
    ...         class_counts = np.array([np.count_nonzero(predicted_labels == _class) for _class in self.classes])
    ...         return class_counts / len(predicted_labels)
    >>> quantifier = MyQuantifier(learner=RandomForestClassifier(), param=1)
    >>> quantifier.get_params(deep=False)
    {'learner': RandomForestClassifier(), 'param': 1}
    >>> # Sample data
    >>> X = np.array([[0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3], 
    ...               [0.5, 0.6], [0.6, 0.5], [0.7, 0.8], [0.8, 0.7], 
    ...               [0.9, 1.0], [1.0, 0.9]])
    >>> y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 1])  # 40% positive (4 out of 10)
    >>> # Split the data into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    >>> # Fit the quantifier
    >>> quantifier.fit(X_train, y_train)
    None
    >>> # Real prevalence in the training set
    >>> get_real_prev(y_train)
    {0: 0.5714285714285714, 1: 0.42857142857142855}
    >>> # Predicted prevalence in the test set
    >>> quantifier.predict(X_test)
    {0: 0.6666666666666666, 1: 0.3333333333333333}

    Example 2: Binary Quantifier
    >>> from sklearn.svm import SVC
    >>> class BinaryQuantifier(AggregativeQuantifier):
    ...     @property
    ...     def is_multiclass(self):
    ...         return False
    ...     def __init__(self, learner):
    ...         self.learner = learner
    ...     def _fit_method(self, X, y):
    ...         self.learner.fit(X, y)
    ...         return self
    ...     def _predict_method(self, X):
    ...         predicted_labels = self.learner.predict(X)
    ...         class_counts = np.array([np.count_nonzero(predicted_labels == _class) for _class in self.classes])
    ...         return class_counts / len(predicted_labels)
    >>> binary_quantifier = BinaryQuantifier(learner=SVC(probability=True))
    >>> # Sample multiclass data
    >>> X = np.array([
    ...     [0.1, 0.2], [0.2, 0.1], [0.3, 0.4], [0.4, 0.3], 
    ...     [0.5, 0.6], [0.6, 0.5], [0.7, 0.8], [0.8, 0.7], 
    ...     [0.9, 1.0], [1.0, 0.9], [1.1, 1.2], [1.2, 1.1], 
    ...     [1.3, 1.4], [1.4, 1.3], [1.5, 1.6], [1.6, 1.5], 
    ...     [1.7, 1.8], [1.8, 1.7], [1.9, 2.0], [2.0, 1.9]
    ... ])
    >>> # Update the labels to include a third class
    >>> y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 2, 2, 0, 1, 0, 1, 0, 1, 2, 2, 0, 1])
    >>> # Split the data into training and testing sets
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    >>> # Fit the binary quantifier
    >>> binary_quantifier.fit(X_train, y_train)
    None
    >>> # Real prevalence in the training set
    >>> get_real_prev(y_test)
    {0: 0.25, 1: 0.5, 2: 0.25}
    >>> preds = binary_quantifier.predict(X_test)
    >>> preds
    {0: 1.0, 1: 0.0, 2: 0.0}
    """
    
    
    def __init__(self, learner):
        self.learner = learner

    def aggregation_type(self):
        return "soft"

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method allows setting parameters for both the model and the learner.
        Parameters that match the model's attributes will be set directly on the model.
        Parameters prefixed wit h 'learner__' will be set on the learner if it exists.
        Parameters:
        -----------
        **params : dict
            Dictionary of parameters to set. Keys can be model attribute names or 
            'learner__' prefixed names for learner parameters.
        Returns:
        --------
        self : Quantifier
            Returns the instance of the quantifier with updated parameters itself.
        """
        
        
        # Model Params
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)

        # Learner Params
        if self.learner is not None:
            learner_params = {k.replace('learner__', ''): v for k, v in params.items() if 'learner__' in k}
            if learner_params:
                self.learner.set_params(**learner_params)
        
        return self