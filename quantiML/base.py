from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator
from copy import deepcopy
import numpy as np


from .utils import parallel, normalize_prevalence

class Quantifier(ABC, BaseEstimator):
    """ Abstract Class for quantifiers, with  """
    
    @abstractmethod
    def fit(self, X, y) -> object: ...
    
    @abstractmethod
    def predict(self, X) -> dict: ...
    
    @property
    def classes(self) -> list:
        return self._classes
    
    @classes.setter
    def classes(self, classes):
        self._classes = sorted(classes)
    
    @property
    def n_class(self) -> list:
        return len(self._classes)
    
    @property
    def multiclass_method(self) -> bool:
        return True

    @property
    def binary_data(self) -> bool:
        return len(self._classes) == 2


class AggregativeQuantifier(Quantifier, ABC):
    
    def __init__(self):
        # Dictionary to hold binary quantifiers for each class.
        self.binary_quantifiers = {}

    def fit(self, X, y, learner_fitted=False, cv_folds: int = 10):
        """Fit the quantifier model.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels.
            learner_fitted (bool, optional): Whether the learner is already fitted. Defaults to False.
            cv_folds (int, optional): Number of cross-validation folds. Defaults to 10.

        Returns:
            self: Fitted quantifier.
        """
        self.classes = np.unique(y)
        if self.binary_data or self.multiclass_method:
            return self._fit_method(X, y, learner_fitted, cv_folds)
        
        # Making one vs all
        self.binary_quantifiers = {class_: deepcopy(self) for class_ in self.classes}
        parallel(self.delayed_fit, self.classes, X, y, learner_fitted, cv_folds)
        
        return self

    def predict(self, X) -> dict:
        """Predict class prevalences for the given data.

        Args:
            X (array-like): Test features.

        Returns:
            dict: Dictionary with class prevalences.
        """
        if self.binary_data or self.multiclass_method:
            prevalences = self._predict_method(X)
            return normalize_prevalence(prevalences, self.classes)
        
        # Making one vs all 
        prevalences = parallel(self.delayed_predict, self.classes, X)
        return normalize_prevalence(prevalences, self.classes)
    
    @abstractmethod
    def _fit_method(self, X, y, learner_fitted: bool, cv_folds: int):
        """Abstract fit method that each quantification method must implement.

        Args:
            X (array-like): Training features.
            y (array-like): Training labels.
            learner_fitted (bool): Whether the learner is already fitted.
            cv_folds (int): Number of cross-validation folds.
        """
        ...

    @abstractmethod
    def _predict_method(self, X) -> dict:
        """Abstract predict method that each quantification method must implement.

        Args:
            X (array-like): Test data to generate class prevalences.

        Returns:
            dict: Dictionary with class:prevalence for each class.
        """
        ...
    
    @property
    def learner(self):
        return self.learner_

    @learner.setter
    def learner(self, value):
        self.learner_ = value
        
    # MULTICLASS METHODS
    
    def delayed_fit(self, class_, X, y, learner_fitted, cv_folds):
        """Delayed fit method for one-vs-all strategy, with parallel running.

        Args:
            class_ (Any): The class for which the model is being fitted.
            X (array-like): Training features.
            y (array-like): Training labels.
            learner_fitted (bool): Whether the learner is already fitted.
            cv_folds (int): Number of cross-validation folds.

        Returns:
            self: Fitted binary quantifier for the given class.
        """
        y_class = (y == class_).astype(int)
        return self.binary_quantifiers[class_]._fit_method(X, y_class, learner_fitted, cv_folds)
    
    def delayed_predict(self, class_, X):
        """Delayed predict method for one-vs-all strategy, with parallel running.

        Args:
            class_ (Any): The class for which the model is making predictions.
            X (array-like): Test features.

        Returns:
            float: Predicted prevalence for the given class.
        """
        return self.binary_quantifiers[class_]._predict_method(X)[1]
