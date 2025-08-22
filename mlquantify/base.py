from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
from copy import deepcopy
import numpy as np
import joblib

import mlquantify as mq
from .utils.general import parallel, normalize_prevalence


class DynamicDomainHandler:

    def __init__(self, *args, **kwargs):
        assert isinstance(self, Quantifier), "DynamicDomainHandler can only be used with Quantifier instances"
        if hasattr(self, 'fit'):
            self._original_fit = self.fit
        if hasattr(self, 'predict'):
            self._original_predict = self.predict

        self.fit = self._handle_fit
        self.predict = self._handle_predict
        self.binary_models = {}

    def _handle_fit(self, X, y, *args, **kwargs):
        self.classes = np.unique(y)
        if len(self.classes) > 2:
            self._fit_ova(X, y, *args, **kwargs)
        self._original_fit(X, y, *args, **kwargs)

    def _fit_ova(self, X, y, *args, **kwargs):
        for _class in self.classes:
            self.binary_models[_class] = deepcopy(self)
            parallel(self.binary_models[_class]._original_fit(X, (y == _class).astype(int), *args, **kwargs))
        
    def _handle_predict(self, X, *args, **kwargs):
        if len(self.classes) > 2:
            return self._predict_ova(X, *args, **kwargs)
        return self._original_predict(X, *args, **kwargs)

    def _predict_ova(self, X, *args, **kwargs):
        predictions = {}
        for _class in self.classes:
            predictions[_class] = self.binary_models[_class]._original_predict(X, *args, **kwargs)
        return predictions


class Quantifier(ABC, BaseEstimator, DynamicDomainHandler):
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
    @handle_domain_fit
    def fit(self, X, y) -> object: ...
    
    @abstractmethod
    @handle_domain_predict
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
        Parameters prefixed with 'learner__' will be set on the learner if it exists.
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