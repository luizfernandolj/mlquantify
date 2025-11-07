from functools import wraps

from mlquantify.utils._validation import _is_fitted
from mlquantify.utils._context import validation_context, is_validation_skipped


def _fit_context(prefer_skip_nested_validation: bool = False):
    """
    Decorator to manage validation context during the fit process.
    
    Parameters
    ----------
    prefer_skip_nested_validation : bool, optional
        If True, prefer to skip nested validation during fitting, by default False.
    """
    def decorator(fit_method):
        @wraps(fit_method)
        def wrapper(estimator, *args, **kwargs):
            global_skip_validation = is_validation_skipped()

            # Avoid validation for partial_fit if already fitted
            partial_fit_and_fitted = (
                fit_method.__name__ == "partial_fit" and _is_fitted(estimator)
            )

            if not global_skip_validation and not partial_fit_and_fitted:
                estimator._validate_params()

            with validation_context(
                skip=(prefer_skip_nested_validation or global_skip_validation)
            ):
                return fit_method(estimator, *args, **kwargs)

        return wrapper

    return decorator
