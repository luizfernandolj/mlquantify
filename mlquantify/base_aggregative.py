from mlquantify.utils._tags import (
    get_tags
)
from mlquantify.utils._validation import validate_parameter_constraints, validate_estimator_constraints
from mlquantify.utils._get_scores import apply_cross_validation
import numpy as np
import inspect


def _get_estimator(quantifier):
    """Return the estimator configured on a quantifier."""
    return getattr(quantifier, "estimator", None)


class AggregativeMixin:
    """Mixin class for all aggregative quantifiers.
    
    An aggregative quantifier is a quantifier that relies on an underlying
    supervised estimator to produce predictions on which the quantification 
    is then performed.
    
    Inheriting from this mixin provides estimator validation and setting 
    parameters also for the estimator (used by `GridSearchQ` and friends).
    
    This mixin also sets the `has_estimator` and `requires_fit`
    tags to `True`.


    Notes
    -----
    - An aggregative quantifier must have a 'estimator' attribute that is
      a supervised learning estimator.
    - Depending on the type of predictions required from the estimator, 
      the quantifier can be further classified as a 'soft' or 'crisp' 
      aggregative quantifier. 
      
    Read more in the :ref:`User Guide <rolling_your_own_aggregative_quantifiers>` 
    for more details.
    
    
    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier
    >>> from mlquantify.base_aggregative import AggregativeMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> class MyAggregativeQuantifier(AggregativeMixin, BaseQuantifier):
    ...     def __init__(self, estimator=None):
    ...         self.estimator = estimator if estimator is not None else LogisticRegression()
    ...     def fit(self, X, y):
    ...         self.estimator.fit(X, y)
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...     def predict(self, X):
    ...         preds = self.estimator.predict(X)
    ...         _, counts = np.unique(preds, return_counts=True)
    ...         prevalence = counts / counts.sum()
    ...         return prevalence
    >>> quantifier = MyAggregativeQuantifier()
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(0, 2, size=100)
    >>> quantifier.fit(X, y).predict(X)
    [0.5 0.5]
    """
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.has_estimator = True
        tags.requires_fit = True
        return tags
    
    
    def _fit_estimator_predictions(
        self,
        X,
        y,
        estimator_fitted=False,
        cv=None,
        stratified=None,
        shuffle=None,
        random_state=None,
        cv_prediction="refit",
    ):
        estimator_function = _get_estimator_function(self)
        estimator = _get_estimator(self)

        if estimator_fitted:
            self.estimator_ = estimator
            return getattr(estimator, estimator_function)(X), y

        cv = getattr(self, "cv", None) if cv is None else cv
        stratified = getattr(self, "stratified", True) if stratified is None else stratified
        shuffle = getattr(self, "shuffle", True) if shuffle is None else shuffle
        random_state = getattr(self, "random_state", None) if random_state is None else random_state

        if cv is None:
            cv = 5

        predictions, y_pred, fitted_estimator = apply_cross_validation(
            estimator,
            X,
            y,
            function=estimator_function,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
            cv_prediction=cv_prediction,
        )
        self.estimator_ = fitted_estimator

        return predictions, y_pred

    def _predict_estimator(self, X):
        estimator_function = _get_estimator_function(self)
        estimator = getattr(self, "estimator_", None)
        if estimator is None:
            estimator = _get_estimator(self)

        if isinstance(estimator, list):
            return self._predict_estimator_ensemble(
                estimator,
                X,
                estimator_function,
            )

        return getattr(estimator, estimator_function)(X)

    def _predict_estimator_ensemble(self, estimators, X, estimator_function):
        if not estimators:
            raise ValueError("The estimator ensemble is empty.")

        if estimator_function == "predict_proba":
            return self._predict_proba_ensemble(estimators, X)

        predictions = [
            getattr(estimator, estimator_function)(X)
            for estimator in estimators
        ]

        if estimator_function == "predict":
            return self._majority_vote(predictions)

        return np.mean(np.asarray(predictions, dtype=float), axis=0)

    def _predict_proba_ensemble(self, estimators, X):
        classes = np.asarray(getattr(self, "classes_", estimators[0].classes_))
        aligned_predictions = []

        for estimator in estimators:
            probabilities = estimator.predict_proba(X)
            estimator_classes = np.asarray(
                getattr(estimator, "classes_", classes)
            )

            if (
                probabilities.ndim != 2
                or probabilities.shape[1] != estimator_classes.size
            ):
                aligned_predictions.append(probabilities)
                continue

            aligned = np.zeros(
                (probabilities.shape[0], classes.size),
                dtype=float,
            )

            for source_idx, cls in enumerate(estimator_classes):
                target_idx = np.flatnonzero(classes == cls)
                if target_idx.size:
                    aligned[:, target_idx[0]] = probabilities[:, source_idx]

            aligned_predictions.append(aligned)

        return np.mean(np.asarray(aligned_predictions, dtype=float), axis=0)

    @staticmethod
    def _majority_vote(predictions):
        predictions = np.asarray(predictions)
        votes = []

        for sample_predictions in predictions.T:
            values, counts = np.unique(
                sample_predictions,
                return_counts=True,
            )
            votes.append(values[np.argmax(counts)])

        return np.asarray(votes)


    def _validate_params(self):
        """Validate the parameters of the quantifier instance,
        including the estimator's parameters.
        
        The expected types and values must be defined in the `_parameter_constraints`
        class attribute as a dictionary. `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for more details.
        """
        validate_estimator_constraints(self, _get_estimator(self))
        
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )

    def get_params(self, deep=True):
        params = super().get_params(deep=deep)

        if not self._accepts_estimator_parameter():
            return params

        estimator = _get_estimator(self)

        params.pop("estimator", None)
        for key in list(params):
            if key.startswith("estimator__"):
                params.pop(key)

        params["estimator"] = estimator

        if deep and estimator is not None and hasattr(estimator, "get_params"):
            for key, value in estimator.get_params(deep=True).items():
                params[f"estimator__{key}"] = value

        return params
    
    def set_params(self, **params):
        estimator_nested = {}
        own_params = {}
        
        for key, value in params.items():
            if key.startswith("estimator__"):
                estimator_nested[key.replace("estimator__", "", 1)] = value
            else:
                own_params[key] = value

        for key, value in own_params.items():
            if key == "estimator":
                self.estimator = value
            elif hasattr(self, key):
                setattr(self, key, value)

        estimator = _get_estimator(self)
        if estimator is not None and estimator_nested:
            estimator.set_params(**estimator_nested)
        
        return self

    def _accepts_estimator_parameter(self):
        try:
            signature = inspect.signature(self.__init__)
        except (TypeError, ValueError):
            return False

        return "estimator" in signature.parameters
    

class SoftPredictionMixin:
    """Soft predictions mixin for aggregative quantifiers.

    This mixin provides the following change tags:
    - `estimator_function`: "predict_proba"
    - `estimator_type`: "soft"
    
    
    Notes
    -----
    - This mixin should be used alongside the `AggregativeMixin`, in 
    the left of it in the inheritance order.
    
    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier
    >>> from mlquantify.base_aggregative import AggregativeMixin, SoftPredictionMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> class MySoftAggregativeQuantifier(SoftPredictionMixin, AggregativeMixin, BaseQuantifier):
    ...     def __init__(self, estimator=None):
    ...         self.estimator = estimator if estimator is not None else LogisticRegression()
    ...     def fit(self, X, y):
    ...         self.estimator.fit(X, y)
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...     def predict(self, X):
    ...         proba = self.estimator.predict_proba(X)
    ...         return proba.mean(axis=0)
    >>> quantifier = MySoftAggregativeQuantifier()
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(0, 2, size=100)
    >>> quantifier.fit(X, y).predict(X)
    [0.5 0.5]
    """
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator_function = "predict_proba"
        tags.estimator_type = "soft"
        return tags


class CrispPredictionMixin:
    """Crisp predictions mixin for aggregative quantifiers.
    
    This mixin provides the following change tags:
    - `estimator_function`: "predict"
    - `estimator_type`: "crisp"
    
    
    Notes
    -----
    - This mixin should be used alongside the `AggregativeMixin`, in
    the left of it in the inheritance order.
    
    
    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier
    >>> from mlquantify.base_aggregative import AggregativeMixin, CrispPredictionMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> class MyCrispAggregativeQuantifier(CrispPredictionMixin, AggregativeMixin, BaseQuantifier):
    ...     def __init__(self, estimator=None):
    ...         self.estimator = estimator if estimator is not None else LogisticRegression()
    ...     def fit(self, X, y):
    ...         self.estimator.fit(X, y)
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...     def predict(self, X):
    ...         preds = self.estimator.predict(X)
    ...         _, counts = np.unique(preds, return_counts=True)
    ...         prevalence = counts / counts.sum()
    ...         return prevalence
    >>> quantifier = MyCrispAggregativeQuantifier()
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(0, 2, size=100)
    >>> quantifier.fit(X, y).predict(X)
    [0.5 0.5]
    """

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator_function = "predict"
        tags.estimator_type= "crisp"
        return tags


AggregationMixin = AggregativeMixin


def uses_soft_predictions(quantifier):
    """Check if the quantifier uses soft predictions."""
    return get_tags(quantifier).estimator_type == "soft"

def uses_crisp_predictions(quantifier):
    """Check if the quantifier uses crisp predictions."""
    return get_tags(quantifier).estimator_type == "crisp"

def is_aggregative_quantifier(quantifier):
    """Check if the quantifier is aggregative."""
    return get_tags(quantifier).has_estimator

def get_aggregation_requirements(quantifier):
    """Get the prediction requirements for the aggregative quantifier."""
    tags = get_tags(quantifier)
    return tags.prediction_requirements


def _get_estimator_function(quantifier):
    """Get the estimator function name used by the aggregative quantifier."""
    tags = get_tags(quantifier)
    function_name = tags.estimator_function
    if function_name is None:
        raise ValueError(f"The quantifier {quantifier.__class__.__name__} does not specify an estimator function.")
    estimator = _get_estimator(quantifier)
    if estimator is None:
        raise AttributeError(
            f"The quantifier {quantifier.__class__.__name__} does not have an estimator configured."
        )
    if not hasattr(estimator, function_name):
        raise AttributeError(f"The estimator {estimator.__class__.__name__} does not have the method '{function_name}'.")
    return function_name
