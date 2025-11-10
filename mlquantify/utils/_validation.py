from __future__ import annotations
from typing import Any
import numpy as np
import scipy.sparse as sp
from sklearn.utils.validation import check_array, check_X_y, _check_y

from mlquantify.utils._tags import TargetInputTags, get_tags
from mlquantify.utils._exceptions import InputValidationError, InvalidParameterError, NotFittedError
from mlquantify.utils._constraints import make_constraint


# ---------------------------
# y Validation
# ---------------------------

def _validate_is_numpy_array(array: Any) -> None:
    """Ensure y are a numpy array."""
    if not isinstance(array, np.ndarray):
        raise InputValidationError(
            f"y must be a numpy array, got {type(y).__name__}."
        )


def _validate_1d_predictions(quantifier: Any, y: np.ndarray, target_tags: TargetInputTags) -> None:
    """Validate 1D predictions according to quantifier tags."""
    if target_tags.continuous:
        return  # continuous allows any numeric vector

    n_class = len(np.unique(y))

    if target_tags.one_d:
        
        if n_class > 2 and not target_tags.multi_class:
            raise InputValidationError(
                f"1D predictions for {quantifier.__class__.__name__} must be binary "
                f"with 2 unique values, got {n_class} unique values."
            )
        if not np.issubdtype(y.dtype, np.number) and not target_tags.categorical:
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
    if target_tags.two_d:
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

def _get_valid_crisp_predictions(predictions, threshold=0.5):
    predictions = np.asarray(predictions)

    dimensions = predictions.shape[1] if len(predictions.shape) > 1 else 1

    if dimensions > 2:
        predictions = np.argmax(predictions, axis=1)
    elif dimensions == 2:
        predictions = (predictions[:, 1] > threshold).astype(int)
    elif dimensions == 1:
        if np.issubdtype(predictions.dtype, np.floating):
            predictions = (predictions > threshold).astype(int)
    else:
        raise ValueError(f"Predictions array has an invalid number of dimensions. Expected 1 or more dimensions, got {predictions.ndim}.")

    return predictions


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
    elif estimator_type == "crisp" and np.issubdtype(predictions.dtype, np.floating):
        predictions = _get_valid_crisp_predictions(predictions) 
    return predictions   
    
    


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


def _is_arraylike_not_scalar(array):
    """Return True if array is array-like and not a scalar"""
    return _is_arraylike(array) and not np.isscalar(array)


def _is_arraylike(x):
    """Returns whether the input is array-like."""
    if sp.issparse(x):
        return False

    return hasattr(x, "__len__") or hasattr(x, "shape") or hasattr(x, "__array__")


def validate_data(quantifier, 
                  X="no_validation",
                  y="no_validation",
                  reset=True,
                  validate_separately=False,
                  skip_check_array=False,
                  **check_params):
    """
    Validate input data X and optionally y using sklearn's validate_data.
    Raises InputValidationError if validation fails.
    """
    tags = get_tags(quantifier)
    if y is None and tags.target_input_tags.required:
        raise InputValidationError(
            f"The target variable y is required for {quantifier.__class__.__name__}."
        )
    
    no_val_X = isinstance(X, str) and X == "no_validation"
    no_val_y = y is None or (isinstance(y, str) and y == "no_validation")
    
    if no_val_X and no_val_y:
        raise ValueError("Validation should be done on X, y or both.")
    
    default_check_params = {"estimator": quantifier}
    check_params = {**default_check_params, **check_params}
    
    if skip_check_array:
        if not no_val_X and no_val_y:
            out = X
        elif no_val_X and not no_val_y:
            out = y
        else:
            out = X, y
    elif not no_val_X and no_val_y:
        out = check_array(X, input_name="X", **check_params)
    elif no_val_X and not no_val_y:
        out = _check_y(y, **check_params)
    else:
        if validate_separately:
            # We need this because some estimators validate X and y
            # separately, and in general, separately calling check_array()
            # on X and y isn't equivalent to just calling check_X_y()
            # :(
            check_X_params, check_y_params = validate_separately
            if "estimator" not in check_X_params:
                check_X_params = {**default_check_params, **check_X_params}
            X = check_array(X, input_name="X", **check_X_params)
            if "estimator" not in check_y_params:
                check_y_params = {**default_check_params, **check_y_params}
            y = check_array(y, input_name="y", **check_y_params)
        else:
            X, y = check_X_y(X, y, **check_params)
        out = X, y
        
    return out


def validate_prevalences(quantifier, prevalences: np.ndarray | list | dict, classes: np.ndarray, return_type: str = "dict", normalize: bool = True) -> dict | np.ndarray:
    """
    Validate class prevalences according to quantifier tags.
    
    Parameters
    ----------
    quantifier : estimator
        The quantifier instance
    prevalences : np.ndarray, list, or dict
        Predicted prevalences for each class
    classes : np.ndarray
        Array of class labels
    return_type : str, default="dict"
        Return format: "dict" or "array"
    normalize : bool, default=True
        Whether to normalize prevalences to sum to 1
        
    Returns
    -------
    dict or np.ndarray
        Validated prevalences in the requested format
    """
    if return_type not in ["dict", "array"]:
        raise InvalidParameterError(
            f"return_type must be 'dict' or 'array', got {return_type!r}."
        )
    
    # Convert to dict if needed
    if isinstance(prevalences, dict):
        prev_dict = prevalences
    elif isinstance(prevalences, (list, np.ndarray)):
        prevalences = np.asarray(prevalences)
        
        if len(prevalences) > len(classes):
            raise InputValidationError(
                f"Number of prevalences ({len(prevalences)}) cannot exceed number of classes ({len(classes)})."
            )
        
        # Create dict, padding with zeros if classes is larger
        prev_dict = {}
        for i, cls in enumerate(classes):
            prev_dict[cls] = prevalences[i] if i < len(prevalences) else 0.0
    else:
        raise InputValidationError(
            f"prevalences must be a numpy array, list, or dict, got {type(prevalences).__name__}."
        )
    
    # Validate all classes are present
    if set(prev_dict.keys()) != set(classes):
        raise InputValidationError(
            f"prevalences keys must match classes. Got keys {set(prev_dict.keys())}, expected {set(classes)}."
        )
    
    # Normalize if requested
    if normalize:
        total = sum(prev_dict.values())
        if total == 0:
            raise InputValidationError("Cannot normalize prevalences: sum is zero.")
        prev_dict = {cls: val / total for cls, val in prev_dict.items()}
    
    # Convert numpy types to native Python types for cleaner output
    
    prev_dict_converted = {}
    # Convert numpy types to native Python types
    for cls, val in prev_dict.items():
        if isinstance(cls, np.integer):
            cls = int(cls)
        elif isinstance(cls, np.floating):
            cls = float(cls)
        elif isinstance(cls, np.str_):
            cls = str(cls)
        prev_dict_converted[cls] = float(val)
    
    # Return in requested format
    if return_type == "dict":
        return prev_dict_converted
    else:
        return np.array([prev_dict_converted[cls] for cls in classes])


def normalize_prevalences(prevalences: np.ndarray | list | dict, classes: np.ndarray = None) -> np.ndarray | dict:
    """
    Normalize prevalences to sum to 1.
    
    Parameters
    ----------
    prevalences : np.ndarray, list, or dict
        Class prevalences to normalize
    classes : np.ndarray, optional
        Array of class labels (required if prevalences is array/list)
        
    Returns
    -------
    np.ndarray or dict
        Normalized prevalences in the same format as input
    """
    if isinstance(prevalences, dict):
        total = sum(prevalences.values())
        if total == 0:
            raise InputValidationError("Cannot normalize prevalences: sum is zero.")
        normalized = {cls: val / total for cls, val in prevalences.items()}
        
        normalized_dict = {}
        # Convert numpy types to native Python types
        for cls, val in normalized.items():
            if isinstance(cls, np.integer):
                cls = int(cls)
            elif isinstance(cls, np.floating):
                cls = float(cls)
            elif isinstance(cls, np.str_):
                cls = str(cls)
            normalized_dict[cls] = float(val)
        return normalized_dict
    
    elif isinstance(prevalences, (list, np.ndarray)):
        prevalences = np.asarray(prevalences)
        total = prevalences.sum()
        if total == 0:
            raise InputValidationError("Cannot normalize prevalences: sum is zero.")
        return prevalences / total
    
    else:
        raise InputValidationError(
            f"prevalences must be a numpy array, list, or dict, got {type(prevalences).__name__}."
        )
        
        
def check_has_method(obj: Any, method_name: str) -> bool:
    """Check if the object has a callable method with the given name."""
    return callable(getattr(obj, method_name, None))

def check_classes_attribute(quantifier: Any, classes) -> bool:
    """Check if the quantifier has a 'classes_' attribute and if it matches the type of classes."""
    
    if not hasattr(quantifier, "classes_"):
        return classes
    
    quantifier_classes = quantifier.classes_
    
    # Check if types match
    if type(quantifier_classes) != type(classes):
        return classes
    
    # Check if shapes match before comparing elements
    if len(quantifier_classes) != len(classes) or not np.all(quantifier_classes == classes):
        return classes
    return quantifier_classes
    