from dataclasses import dataclass, field


@dataclass
class TargetInputTags:
    """Describes the accepted shape and type of test-time input.

    Attributes
    ----------
    one_d : bool, default=True
        Whether the quantifier accepts 1-D input (hard labels or binary
        posterior scores).
    two_d : bool, default=False
        Whether the quantifier accepts a 2-D posterior probability matrix.
    continuous : bool, default=False
        If ``True``, continuous-valued inputs are permitted.
    categorical : bool, default=True
        If ``True``, categorical class labels are permitted.
    multi_class : bool, default=True
        If ``True``, more than two classes are supported.
    required : bool, default=False
        If ``True``, a target array is required by ``fit``.
    """

    one_d: bool = True
    two_d: bool = False
    continuous: bool = False
    categorical: bool = True
    multi_class: bool = True
    required: bool = False

@dataclass
class PredictionRequirements:
    """Specifies which arrays must be passed to ``aggregate``.

    Attributes
    ----------
    requires_train_proba : bool, default=True
        Whether training posterior probabilities are needed.
    requires_train_labels : bool, default=True
        Whether training class labels are needed.
    requires_test_predictions : bool, default=True
        Whether test predictions are needed.
    """

    requires_train_proba: bool = True
    requires_train_labels: bool = True
    requires_test_predictions: bool = True


@dataclass
class Tags:
    """Capability descriptor for a quantifier.

    Attributes
    ----------
    estimation_type : str or None
        High-level estimation type (e.g. ``'aggregative'``, ``'sample'``).
    estimator_function : str or None
        Name of the classifier method used to produce predictions
        (e.g. ``'predict_proba'``, ``'predict'``).
    estimator_type : str or None
        Output type expected from the classifier (``'soft'`` or ``'crisp'``).
    aggregation_type : str or None
        Aggregation strategy identifier.
    target_input_tags : TargetInputTags
        Tags describing accepted input shapes and types.
    prediction_requirements : PredictionRequirements
        Tags describing which arrays are needed for aggregation.
    has_estimator : bool, default=False
        Whether the quantifier wraps a classifier.
    requires_fit : bool, default=True
        Whether the quantifier must be fitted before ``predict``.
    """

    estimation_type: str | None
    estimator_function: str | None
    estimator_type: str | None
    aggregation_type: str | None
    target_input_tags: TargetInputTags
    prediction_requirements: PredictionRequirements
    has_estimator: bool = False
    requires_fit: bool = True



def get_tags(quantifier):
    """Retrieve the :class:`Tags` object for a quantifier.

    Calls ``quantifier.__mlquantify_tags__()`` and returns the result.

    Parameters
    ----------
    quantifier : BaseQuantifier
        Any quantifier that inherits from
        :class:`~mlquantify.base.BaseQuantifier`.

    Returns
    -------
    tags : Tags
        Capability descriptor for the quantifier.

    Raises
    ------
    AttributeError
        If the quantifier does not implement ``__mlquantify_tags__``.
    """
    try:
        tags = quantifier.__mlquantify_tags__()
    except AttributeError as ext:
        if "has no attribute '__mlquantify_tags__'" in str(ext):
            raise AttributeError("Quantifier is missing __mlquantify_tags__ method, ensure your class is inheriting from BaseQuantifier.")
        else:
            raise
    return tags