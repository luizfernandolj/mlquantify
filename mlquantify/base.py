from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator

from mlquantify.utils._tags import (
    PredictionRequirements,
    Tags,
    TargetInputTags,
    get_tags
)
from mlquantify.utils._validation import validate_parameter_constraints



class BaseQuantifier(ABC, BaseEstimator):
   
    
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
            prediction_requirements=PredictionRequirements(),
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
    
