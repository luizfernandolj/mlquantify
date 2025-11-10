from mlquantify.utils._tags import (
    get_tags
)
from mlquantify.utils._validation import validate_parameter_constraints, validate_learner_contraints


class AggregationMixin:
    """Mixin class for all aggregative quantifiers.
    
    An aggregative quantifier is a quantifier that relies on an underlying
    supervised learner to produce predictions on which the quantification 
    is then performed.
    
    Inheriting from this mixin provides learner validation and setting 
    parameters also for the learner (used by `GridSearchQ` and friends).
    
    This mixin also sets the `has_estimator` and `requires_fit`
    tags to `True`.


    Notes
    -----
    - An aggregative quantifier must have a 'learner' attribute that is
      a supervised learning estimator.
    - Depending on the type of predictions required from the learner, 
      the quantifier can be further classified as a 'soft' or 'crisp' 
      aggregative quantifier. 
      
    Read more in the :ref:`User Guide <rolling_your_own_aggregative_quantifiers>` 
    for more details.
    
    
    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier, AggregationMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> class MyAggregativeQuantifier(AggregationMixin, BaseQuantifier):
    ...     def __init__(self, learner=None):
    ...         self.learner = learner if learner is not None else LogisticRegression()
    ...     def fit(self, X, y):
    ...         self.learner.fit(X, y)
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...     def predict(self, X):
    ...         preds = self.learner.predict(X)
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


    def _validate_params(self):
        """Validate the parameters of the quantifier instance,
        including the learner's parameters.
        
        The expected types and values must be defined in the `_parameter_constraints`
        class attribute as a dictionary. `param_name: list of constraints`. See
        the docstring of `validate_parameter_constraints` for more details.
        """
        validate_learner_contraints(self, self.learner)
        
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )
    
    def set_params(self, **params):
        
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
    

class SoftLearnerQMixin:
    """Soft predictions mixin for aggregative quantifiers.

    This mixin provides the following change tags:
    - `estimator_function`: "predict_proba"
    - `estimator_type`: "soft"
    
    
    Notes
    -----
    - This mixin should be used alongside the `AggregationMixin`, in 
    the left of it in the inheritance order.
    
    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier, AggregationMixin, SoftLearnerQMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> class MySoftAggregativeQuantifier(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    ...     def __init__(self, learner=None):
    ...         self.learner = learner if learner is not None else LogisticRegression()
    ...     def fit(self, X, y):
    ...         self.learner.fit(X, y)
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...     def predict(self, X):
    ...         proba = self.learner.predict_proba(X)
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


class CrispLearnerQMixin:
    """Crisp predictions mixin for aggregative quantifiers.
    
    This mixin provides the following change tags:
    - `estimator_function`: "predict"
    - `estimator_type`: "crisp"
    
    
    Notes
    -----
    - This mixin should be used alongside the `AggregationMixin`, in
    the left of it in the inheritance order.
    
    
    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier, AggregationMixin, CrispLearnerQMixin
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> class MyCrispAggregativeQuantifier(CrispLearnerQMixin, AggregationMixin, BaseQuantifier):
    ...     def __init__(self, learner=None):
    ...         self.learner = learner if learner is not None else LogisticRegression()
    ...     def fit(self, X, y):
    ...         self.learner.fit(X, y)
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...     def predict(self, X):
    ...         preds = self.learner.predict(X)
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


def _get_learner_function(quantifier):
    """Get the learner function name used by the aggregative quantifier."""
    tags = get_tags(quantifier)
    function_name = tags.estimator_function
    if function_name is None:
        raise ValueError(f"The quantifier {quantifier.__class__.__name__} does not specify an estimator function.")
    if not hasattr(quantifier.learner, function_name):
        raise AttributeError(f"The learner {quantifier.learner.__class__.__name__} does not have the method '{function_name}'.")
    return function_name