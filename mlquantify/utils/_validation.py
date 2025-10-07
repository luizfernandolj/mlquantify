from __future__ import annotations
from typing import Any
import numpy as np

from mlquantify.utils._tags import TargetInputTags, get_tags
from mlquantify.utils._exceptions import InputValidationError, InvalidParameterError, NotFittedError
from mlquantify.utils._constraints import make_constraint


# ---------------------------
# y Validation
# ---------------------------

def _validate_is_numpy_array(y: Any) -> None:
    """Ensure y are a numpy array."""
    if not isinstance(y, np.ndarray):
        raise InputValidationError(
            f"y must be a numpy array, got {type(y).__name__}."
        )


def _validate_1d_predictions(quantifier: Any, y: np.ndarray, target_tags: TargetInputTags) -> None:
    """Validate 1D predictions according to quantifier tags."""
    if target_tags.continuous:
        return  # continuous allows any numeric vector

    if target_tags.one_d and not target_tags.multi_class:
        if not np.issubdtype(y.dtype, np.number):
            raise InputValidationError(
                f"1D predictions for {quantifier.__class__.__name__} must be numeric (int or float), "
                f"got dtype {y.dtype}."
            )
        return

    if not target_tags.one_d:
        raise InputValidationError(
            f"{quantifier.__class__.__name__} does not accept 1D input according to its tags."
        )


def _validate_2d_predictions(quantifier: Any, y: np.ndarray, target_tags: TargetInputTags) -> None:
    """Validate 2D predictions according to quantifier tags."""
    if not (target_tags.two_d or target_tags.multi_class):
        raise InputValidationError(
            f"{quantifier.__class__.__name__} does not accept multi-class or 2D input."
        )

    if not np.issubdtype(y.dtype, np.floating):
        raise InputValidationError(
            f"{quantifier.__class__.__name__} expects float probabilities for 2D predictions, "
            f"got dtype {y.dtype}."
        )

    # Efficient normalization check for soft probabilities
    if target_tags.multi_class:
        row_sums = y.sum(axis=1)
        if np.abs(row_sums - 1).max() > 1e-3:
            raise InputValidationError(
                f"Soft predictions for multiclass quantifiers must sum to 1 across columns "
                f"(max deviation={np.abs(row_sums - 1).max():.3g})."
            )


def validate_y(quantifier: Any, y: np.ndarray) -> None:
    """
    Validate predictions using the quantifier's declared input tags.
    Raises InputValidationError if inconsistent with tags.
    """
    _validate_is_numpy_array(y)

    try:
        tags = get_tags(quantifier)
        target_tags = tags.target_input_tags
        estimator_type = tags.estimator_type
    except AttributeError as e:
        raise InputValidationError(
            f"Quantifier {quantifier.__class__.__name__} does not implement __mlquantify_tags__()."
        ) from e
    
    if y.ndim == 1:
        _validate_1d_predictions(quantifier, y, target_tags)
    elif y.ndim == 2:
        _validate_2d_predictions(quantifier, y, target_tags)
    else:
        raise InputValidationError(
            f"Predictions must be 1D or 2D array, got array with ndim={y.ndim} and shape={y.shape}."
        )


def validate_predictions(quantifier: Any, predictions: np.ndarray) -> None:
    """
    Validate predictions using the quantifier's declared output tags.
    Raises InputValidationError if inconsistent with tags.
    """
    _validate_is_numpy_array(predictions)

    try:
        tags = get_tags(quantifier)
        estimator_type = tags.estimator_type
    except AttributeError as e:
        raise InputValidationError(
            f"Quantifier {quantifier.__class__.__name__} does not implement __mlquantify_tags__()."
        ) from e

    if estimator_type == "soft" and np.issubdtype(predictions.dtype, np.integer):
        raise InputValidationError(
            f"Soft predictions for {quantifier.__class__.__name__} must be float, got dtype {predictions.dtype}."
        )    
    
    


# ---------------------------
# Parameter Validation
# ---------------------------

def validate_parameter_constraints(parameter_constraints: dict[str, Any], params: dict[str, Any], caller_name: str) -> None:
    """Validate parameters against their declared constraints."""
    for param_name, param_val in params.items():
        if param_name not in parameter_constraints:
            continue

        constraints = parameter_constraints[param_name]
        if constraints == "no_validation":
            continue

        constraint_objs = [make_constraint(c) for c in constraints]

        if any(c.is_satisfied_by(param_val) for c in constraint_objs):
            continue  # valid parameter

        # Only visible constraints in error message
        visible = [c for c in constraint_objs if not getattr(c, "hidden", False)] or constraint_objs
        constraint_str = (
            str(visible[0])
            if len(visible) == 1
            else ", ".join(map(str, visible[:-1])) + f" or {visible[-1]}"
        )

        raise InvalidParameterError(
            f"The parameter '{param_name}' of {caller_name} must be {constraint_str}. "
            f"Got {param_val!r} (type={type(param_val).__name__})."
        )
        
        
def validate_learner_contraints(quantifier, learner) -> None:
    """Validate the learner parameter of a quantifier."""
    try:
        tags = get_tags(quantifier)
    except AttributeError as e:
        raise InvalidParameterError(
            f"Quantifier {quantifier.__class__.__name__} does not implement __mlquantify_tags__()."
        ) from e

    if not tags.has_estimator:
        if learner is not None:
            raise InvalidParameterError(
                f"The quantifier {quantifier.__class__.__name__} does not support using a learner."
            )
        return  # No learner needed


    estimator_function = tags.estimator_function
    
    if estimator_function is None:
        raise InvalidParameterError(f"The quantifier {quantifier.__class__.__name__} does not specify a valid estimator_function in its tags.")
    elif estimator_function == "predict":
        if not hasattr(quantifier.learner, "predict"):
            raise InvalidParameterError(f"The provided learner does not have a 'predict' method, which is required by the quantifier {quantifier.__class__.__name__}.")
    elif estimator_function == "predict_proba":
        if not hasattr(quantifier.learner, "predict_proba"):
            raise InvalidParameterError(f"The provided learner does not have a 'predict_proba' method, which is required by the quantifier {quantifier.__class__.__name__}.")


def _is_fitted(quantifier, attributes=None, all_or_any=all):
    """Check if the quantifier is fitted by verifying the presence of specified attributes."""
    if attributes is None:
        attributes = ["is_fitted_"]

    checks = [hasattr(quantifier, attr) for attr in attributes]
    return all(checks) if all_or_any == all else any(checks)


def check_is_fitted(quantifier, attributes=None, *, msg=None, all_or_any=all):
    """Raise NotFittedError if the quantifier is not fitted."""

    if msg is None:
        msg = f"This {quantifier.__class__.__name__} instance is not fitted yet. Call 'fit' first."

    if not hasattr(quantifier, "fit"):
        raise TypeError(f"Cannot check if {quantifier.__class__.__name__} is fitted: no 'fit' method found.")
    
    tags = get_tags(quantifier)
    
    if not tags.requires_fit and attributes is None:
        return  # No fitting required for this quantifier
    
    if not _is_fitted(quantifier, attributes, all_or_any):
        raise NotFittedError(msg % {"name": type(quantifier).__name__})
