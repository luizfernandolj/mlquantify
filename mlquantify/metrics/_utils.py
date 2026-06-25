import numpy as np


def process_inputs(prev_real, prev_pred):
    """Normalise a pair of prevalence vectors for metric computation.

    .. :noindex:

    Shared helper for the metric implementations. It coerces ``dict`` / ``list``
    inputs to :class:`numpy.ndarray` and zero-pads the shorter vector so both
    have the same length. The argument order follows the scikit-learn
    convention ``(y_true, y_pred)``.

    Parameters
    ----------
    prev_real : array-like or dict
        True prevalence values for each class.
    prev_pred : array-like or dict
        Predicted prevalence values for each class.

    Returns
    -------
    prev_real, prev_pred : numpy.ndarray
        The two prevalence vectors as arrays of equal length.
    """
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    if isinstance(prev_real, list):
        prev_real = np.asarray(prev_real)
    if isinstance(prev_pred, list):
        prev_pred = np.asarray(prev_pred)

    # Pad with zeros if lengths differ
    len_real = len(prev_real)
    len_pred = len(prev_pred)

    if len_real > len_pred:
        prev_pred = np.pad(prev_pred, (0, len_real - len_pred), constant_values=0)
    elif len_pred > len_real:
        prev_real = np.pad(prev_real, (0, len_pred - len_real), constant_values=0)

    return prev_real, prev_pred
