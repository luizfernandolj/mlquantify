from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator

from mlquantify.utils._tags import (
    Tags,
    TargetInputTags,
    get_tags
)
from mlquantify.utils._decorators import (
    set_binary_method
)
from mlquantify.utils._validation import validate_parameter_constraints, validate_learner_contraints
from mlquantify.utils._constraints import Options



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



# ==================================================== #
# ====================== Mixins ====================== #
# ==================================================== #

class BinaryQMixin:

    _parameter_constraints = {
        "strategy": [Options(["ova", "ovo"])],
    }

    @abstractmethod
    def __init__(self, strategy="ova", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strategy = strategy

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.target_input_tags = TargetInputTags(multi_class=False)
        return tags

    @set_binary_method
    def fit(self, X, y, *args, **kwargs):
        super().fit(X, y, *args, **kwargs)
        return self

    @set_binary_method
    def predict(self, X, *args, **kwargs):
        return super().predict(X, *args, **kwargs)


class SoftLearnerQMixin:
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator_function = "predict_proba"
        tags.estimator_type = "soft"
        return tags


class CrispLearnerQMixin:

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimator_function = "predict"
        tags.estimator_type= "crisp"
        return tags
    

class RegressorQMixin:
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__() 
        tags.estimator_function = "predict"
        tags.estimator_type= "regression"
        return tags


class DistributionMixin:

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimation_type = "distribution"
        return tags


class MaximumLikelihoodMixin:

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimation_type = "likelihood"
        return tags


class ThresholdAdjustmentMixin:

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.estimation_type = "adjusting"
        return tags
    


class AggregativeQuantifierMixin:
    """Base class for all aggregative quantifiers.
    """
    
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.has_estimator = True
        tags.requires_fit = True
        return tags


    def _validate_params(self):
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




def uses_soft_predictions(quantifier):
    return get_tags(quantifier).estimator_type == "soft"

def uses_crisp_predictions(quantifier):
    return get_tags(quantifier).estimator_type == "crisp"

def _get_learner_function(quantifier):
    tags = get_tags(quantifier)
    function_name = tags.estimator_function
    if function_name is None:
        raise ValueError(f"The quantifier {quantifier.__class__.__name__} does not specify an estimator function.")
    if not hasattr(quantifier.learner, function_name):
        raise AttributeError(f"The learner {quantifier.learner.__class__.__name__} does not have the method '{function_name}'.")
    return getattr(quantifier.learner, function_name)