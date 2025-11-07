from abc import ABC
from sklearn.base import BaseEstimator

from mlquantify.utils._tags import (
    PredictionRequirements,
    Tags,
    TargetInputTags,
)
from mlquantify.utils._validation import validate_parameter_constraints



class BaseQuantifier(ABC, BaseEstimator):
    """Base class for all quantifiers in mlquantify.
    
    Inhering from this class provides default implementations for
    
    - setting and getting parameters used by `GridSearchQ` and friends;
    - saving/loading quantifier instances;
    - parameter validation.
    
    Read more in :ref:`User Guide <rolling_your_own_quantifier>`.
    
    
    Notes
    -----
    All quantifiers should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword arguments.
    (No `*args` or `**kwargs` allowed.)
    
    
    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier
    >>> import numpy as np
    >>> class MyQuantifier(BaseQuantifier):
    ...     def __init__(self, param1=42, param2='default'):
    ...         self.param1 = param1
    ...         self.param2 = param2
    ...     def fit(self, X, y):
    ...         self.classes_ = np.unique(y)
    ...         return self
    ...     def predict(self, X):
    ...         _, counts = np.unique(self.classes_, return_counts=True)
    ...         prevalence = counts / counts.sum()
    ...         return prevalence
    >>> quantifier = MyQuantifier(param1=10, param2='custom')
    >>> quantifier.get_params()
    {'param1': 10, 'param2': 'custom'}
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(0, 2, size=100)
    >>> quantifier.fit(X, y).predict(X)
    [0.5 0.5]
    """
   
    
    _parameter_constraints: dict[str, list] = {}
    skip_validation: bool = False

    def _validate_params(self):
        """Validate the parameters of the quantifier instance.
        
        The expected types and values must be defined in the `_parameter_constraints`
        class attribute as a dictionary. `param_name: list of constraints`. See 
        the docstring of `validate_parameter_constraints` for more details.
        
        """
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )

        
    def __mlquantify_tags__(self):
        return Tags(
            has_estimator=None,
            estimation_type=None,
            estimator_function=None,
            estimator_type=None,
            aggregation_type=None,
            target_input_tags=TargetInputTags(),
            prediction_requirements=PredictionRequirements(),
            requires_fit= True
        )
        
    def save_quantifier(self, path: str=None) -> None:
        """Save the quantifier instance to a file."""
        if not path:
            path = f"{self.__class__.__name__}.joblib"
        import joblib
        joblib.dump(self, path)



# ==================================================== #
# ====================== Mixins ====================== #
# ==================================================== #
    

class MetaquantifierMixin:
    """Mixin class for meta-quantifiers.
    
    This mixin is empty, and only exists to indicate that the quantifier is 
    a meta quantifier
    
    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier, MetaquantifierMixin
    >>> from mlquantify.adjust_counting import CC
    >>> class MyMetaQuantifier(MetaquantifierMixin, BaseQuantifier):
    ...     def __init__(self, quantifier=None):
    ...         self.quantifier = quantifier
    ...     def fit(self, X, y):
    ...         if self.quantifier is not None:
    ...             self.quantifier.fit(X, y)
    ...         else:
    ...             self.quantifier = CC()
    ...         return self
    >>> X = np.random.rand(100, 10)
    >>> y = np.random.randint(0, 2, size=100)
    >>> meta_qtf = MyMetaQuantifier().fit(X, y)
    >>> meta_qtf.quantifier
    CC()
    """
    ...
    

class ProtocolMixin:
    """Mixin class for protocol-based quantifiers.
    
    This mixin indicates that the quantifier follows a specific protocol,
    by setting the estimation_type tag to "sample" and requires_fit to False.

    Examples
    --------
    >>> from mlquantify.base import BaseQuantifier, ProtocolMixin
    >>> class MyProtocolQuantifier(ProtocolMixin, BaseQuantifier):
    ...     def __init__(self, param=None):
    ...         self.param = param
    ...     def sample_method(self, X):
    ...         indexes = np.random.choice(len(X), size=10, replace=False)
    ...         X_sample = X[indexes]
    ...         return X_sample
    >>> X = np.random.rand(100, 10)
    >>> protocol_qtf = MyProtocolQuantifier(param=5)
    >>> X_sample = protocol_qtf.sample_method(X)
    >>> X_sample.shape
    (10, 10)
    """
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimation_type = "sample"
        tags.requires_fit = False
        return tags
    
