from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator

from mlquantify.utils._tags import (
    Tags,
    TargetInputTags,
    get_tags
)
from mlquantify.utils._validation import validate_parameter_constraints



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
    
    _parameter_constraints: dict[str, list] = {}
    skip_validation: bool = False

    def _validate_params(self):
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
            requires_fit= True
        )
        
    def save_quantifier(self, path: str=None) -> None:
        if not path:
            path = f"{self.__class__.__name__}.joblib"
        import joblib
        joblib.dump(self, path)



# ==================================================== #
# ====================== Mixins ====================== #
# ==================================================== #
    

class MetaquantifierMixin:
    ...
    

class ProtocolMixin:
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimation_type = "sample"
        tags.requires_fit = False
        return tags
    
