from mlquantify.utils._tags import (
    get_tags
)
from mlquantify.utils._validation import validate_parameter_constraints, validate_learner_contraints


class AggregationMixin:
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
    return function_name