from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
from copy import deepcopy
import numpy as np
import joblib

import mlquantify as mq
from .utils.general import parallel, normalize_prevalence

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
    
    @abstractmethod
    def fit(self, X, y) -> object: ...
    
    @abstractmethod
    def predict(self, X) -> dict: ...
    
    @property
    def classes(self) -> list:
        return self._classes
    
    @classes.setter
    def classes(self, classes):
        self._classes = sorted(list(classes))
    
    @property
    def n_class(self) -> list:
        return len(self._classes)
    
    @property
    def is_multiclass(self) -> bool:
        return True

    @property
    def binary_data(self) -> bool:
        return len(self._classes) == 2
    
    
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
    
    
    def __init__(self):
        # Dictionary to hold binary quantifiers for each class.
        self.binary_quantifiers = {}
        self.learner_fitted = False
        self.cv_folds = 10

    def fit(self, X, y, learner_fitted=False, cv_folds: int = 10, n_jobs:int=1):
        """Fit the quantifier model.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
        learner_fitted : bool, default=False
            Whether the learner is already fitted.
        cv_folds : int, default=10
            Number of cross-validation folds.
        n_jobs : int, default=1
            Number of parallel jobs to run.


        Returns
        -------
        self : object
            The fitted quantifier instance.


        Notes
        -----
        The model dynamically determines whether to perform one-vs-all classification or 
        to directly fit the data based on the type of the problem:
        - If the data is binary or inherently multiclass, the model fits directly using 
          `_fit_method` without creating binary quantifiers.
        - For other cases, the model creates one binary quantifier per class using the 
          one-vs-all approach, allowing for dynamic prediction based on the provided dataset.
        """

        self.n_jobs = n_jobs
        self.learner_fitted = learner_fitted
        self.cv_folds = cv_folds
        
        self.classes = np.unique(y)
        
        if self.binary_data or self.is_multiclass:
            return self._fit_method(X, y)
        
        # Making one vs all
        self.binary_quantifiers = {class_: deepcopy(self) for class_ in self.classes}
        parallel(self.delayed_fit, self.classes, self.n_jobs, X, y)
        
        return self

    def predict(self, X) -> dict:
        """Predict class prevalences for the given data.

        Parameters
        ----------
        X : array-like
            Test features.

        Returns
        -------
        dict
            A dictionary where keys are class labels and values are their predicted prevalences.

        Notes
        -----
        The prediction approach is dynamically chosen based on the data type:
        - For binary or inherently multiclass data, the model uses `_predict_method` to directly 
          estimate class prevalences.
        - For other cases, the model performs one-vs-all prediction, where each binary quantifier 
          estimates the prevalence of its respective class. The results are then normalized to 
          ensure they form valid proportions.
        """

        if self.binary_data or self.is_multiclass:
            prevalences = self._predict_method(X)
            return normalize_prevalence(prevalences, self.classes)
        
        # Making one vs all 
        prevalences = np.asarray(parallel(self.delayed_predict, self.classes, self.n_jobs, X))
        return normalize_prevalence(prevalences, self.classes)
    
    @abstractmethod
    def _fit_method(self, X, y):
        """Abstract fit method that each aggregative quantification method must implement.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
        """
        ...

    @abstractmethod
    def _predict_method(self, X) -> dict:
        """Abstract predict method that each aggregative quantification method must implement.

        Parameters
        ----------
        X : array-like
            Test data to generate class prevalences.

        Returns
        -------
        dict, list, or numpy array
            The predicted prevalences, which can be a dictionary where keys are class labels 
            and values are their predicted prevalences, a list, or a numpy array.
        """

        ...
    
    @property
    def is_probabilistic(self) -> bool:
        """Check if the learner is probabilistic or not.
        
        Returns
        -------
        bool
            True if the learner is probabilistic, False otherwise.
        """
        return False
    
    
    @property
    def learner(self):
        """Returns the learner_ object.
        Returns
        -------
        learner_ : object
            The learner_ object stored in the class instance.
        """
        return self.learner_

    @learner.setter
    def learner(self, value):
        """
        Sets the learner attribute.
        Parameters:
        value : any
            The value to be assigned to the learner_ attribute.
        """
        assert isinstance(value, BaseEstimator) or mq.ARGUMENTS_SETTED, "learner object is not an estimator, or you may change ARGUMENTS_SETTED to True"
        self.learner_ = value
    
    def fit_learner(self, X, y):
        """Fit the learner to the training data.
        
        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
        """
        if self.learner is not None:
            if not self.learner_fitted:
                self.learner_.fit(X, y)
        elif mq.ARGUMENTS_SETTED:
            if self.is_probabilistic and mq.arguments["posteriors_test"] is not None:
                return
            elif not self.is_probabilistic and mq.arguments["y_pred"] is not None:
                return

    def predict_learner(self, X):
        """Predict the class labels or probabilities for the given data.
        
        Parameters
        ----------
        X : array-like
            Test features.
        
        Returns
        -------
        array-like
            The predicted class labels or probabilities.
        """
        if self.learner is not None:
            if self.is_probabilistic:
                return self.learner_.predict_proba(X)
            return self.learner_.predict(X)
        else:
            if mq.ARGUMENTS_SETTED:
                if self.is_probabilistic:
                    return mq.arguments["posteriors_test"]
                return mq.arguments["y_pred"]
            else:
                raise ValueError("No learner object was set and no arguments were setted")

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
    
        
    # MULTICLASS METHODS
    
    def delayed_fit(self, class_, X, y):
        """Delayed fit method for one-vs-all strategy, with parallel execution.

        Parameters
        ----------
        class_ : Any
            The class for which the model is being fitted.
        X : array-like
            Training features.
        y : array-like
            Training labels.

        Returns
        -------
        self : object
            Fitted binary quantifier for the given class.
        """

        y_class = (y == class_).astype(int)
        return self.binary_quantifiers[class_].fit(X, y_class)
    
    def delayed_predict(self, class_, X):
        """Delayed predict method for one-vs-all strategy, with parallel execution.

        Parameters
        ----------
        class_ : Any
            The class for which the model is making predictions.
        X : array-like
            Test features.

        Returns
        -------
        float
            Predicted prevalence for the given class.
        """

        return self.binary_quantifiers[class_].predict(X)[1]


class NonAggregativeQuantifier(Quantifier):
    """Abstract base class for non-aggregative quantifiers.
    
    Non-aggregative quantifiers differ from aggregative quantifiers as they do not use 
    an underlying classifier or specific learner for their predictions.
    
    This class defines the general structure and behavior for non-aggregative quantifiers, 
    including support for multiclass data and dynamic handling of binary and multiclass problems.

    Notes
    -----
    This class requires implementing the `_fit_method` and `_predict_method` in subclasses 
    to define how the quantification is performed. These methods handle the core logic for 
    fitting and predicting class prevalences.

    Examples
    --------
    >>> from myquantify.base import NonAggregativeQuantifier
    >>> import numpy as np
    >>> class MyNonAggregativeQuantifier(NonAggregativeQuantifier):
    ...     def _fit_method(self, X, y):
    ...         # Custom logic for fitting
    ...         pass
    ...     def _predict_method(self, X):
    ...         # Custom logic for predicting
    ...         return {0: 0.5, 1: 0.5}
    >>> quantifier = MyNonAggregativeQuantifier()
    >>> X = np.random.rand(10, 2)
    >>> y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    >>> quantifier.fit(X, y)
    <MyNonAggregativeQuantifier>
    >>> quantifier.predict(X)
    {0: 0.5, 1: 0.5}
    """

    def fit(self, X, y, n_jobs: int = 1):
        """Fit the quantifier model to the training data.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
        n_jobs : int, default=1
            Number of parallel jobs to run.

        Returns
        -------
        self : NonAggregativeQuantifier
            The fitted quantifier instance.

        Notes
        -----
        - For binary or inherently multiclass data, the model directly calls `_fit_method` 
          to process the data.
        - For other cases, it creates one quantifier per class using a one-vs-all strategy 
          and fits each quantifier independently in parallel.
        """
        self.n_jobs = n_jobs
        self.classes = np.unique(y)
        if self.binary_data or self.is_multiclass:
            return self._fit_method(X, y)

        # One-vs-all approach
        self.binary_quantifiers = {class_: deepcopy(self) for class_ in self.classes}
        parallel(self.delayed_fit, self.classes, self.n_jobs, X, y)
        return self

    def predict(self, X) -> dict:
        """Predict class prevalences for the given data.

        Parameters
        ----------
        X : array-like
            Test features.

        Returns
        -------
        dict
            A dictionary where keys are class labels and values are their predicted prevalences.

        Notes
        -----
        - For binary or inherently multiclass data, the model directly calls `_predict_method`.
        - For other cases, it performs one-vs-all prediction, combining the results into a normalized 
          dictionary of class prevalences.
        """
        if self.binary_data or self.is_multiclass:
            prevalences = self._predict_method(X)
            return normalize_prevalence(prevalences, self.classes)

        # One-vs-all approach
        prevalences = np.asarray(parallel(self.delayed_predict, self.classes, self.n_jobs, X))
        return normalize_prevalence(prevalences, self.classes)

    @abstractmethod
    def _fit_method(self, X, y):
        """Abstract method for fitting the quantifier.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.

        Notes
        -----
        This method must be implemented in subclasses to define the fitting logic for 
        the non-aggregative quantifier.
        """
        ...

    @abstractmethod
    def _predict_method(self, X) -> dict:
        """Abstract method for predicting class prevalences.

        Parameters
        ----------
        X : array-like
            Test features.

        Returns
        -------
        dict, list, or numpy array
            The predicted prevalences, which can be a dictionary where keys are class labels 
            and values are their predicted prevalences, a list, or a numpy array.

        Notes
        -----
        This method must be implemented in subclasses to define the prediction logic for 
        the non-aggregative quantifier.
        """
        ...
