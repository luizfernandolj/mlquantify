import numpy as np

def process_inputs(prev_pred, prev_real):
    """
    .. :noindex:
    
    Process the input data for internal use.
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


def NMD(prev_pred, prev_real, distances=None):
    """
    Compute the Normalized Match Distance (NMD), also known as Earth Mover’s Distance (EMD),
    for ordinal quantification evaluation.

    Parameters
    ----------
    prev_real : array-like or dict
        True prevalence values for each ordered class.

    prev_pred : array-like or dict
        Predicted prevalence values for each ordered class.

    distances : array-like of shape (n_classes-1,), optional
        Distance between consecutive classes (d(y_i, y_{i+1})).
        If None, all distances are assumed to be 1.

    Returns
    -------
    nmd : float
        Normalized Match Distance between predicted and true prevalences.
    """
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    n_classes = len(prev_real)

    if distances is None:
        distances = np.ones(n_classes - 1)
    else:
        distances = np.asarray(distances, dtype=float)
        if len(distances) != n_classes - 1:
            raise ValueError("Length of distances must be n_classes - 1.")

    # cumulative differences
    cum_diffs = np.cumsum(prev_pred - prev_real)
    nmd = np.sum(distances * np.abs(cum_diffs[:-1])) / (n_classes - 1)
    return float(nmd)


def RNOD(prev_pred, prev_real, distances=None):
    """
    Compute the Root Normalised Order-aware Divergence (RNOD) for ordinal quantification evaluation.

    Parameters
    ----------
    prev_real : array-like or dict
        True prevalence values for each ordered class.

    prev_pred : array-like or dict
        Predicted prevalence values for each ordered class.

    distances : 2D array-like of shape (n_classes, n_classes), optional
        Distance matrix between classes (d(y_i, y_j)).
        If None, assumes d(y_i, y_j) = |i - j|.

    Returns
    -------
    rnod : float
        Root Normalised Order-aware Divergence between predicted and true prevalences.
    """
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    n_classes = len(prev_real)
    Y_star = np.where(prev_real > 0)[0]

    # default distance: |i - j|
    if distances is None:
        distances = np.abs(np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :])
    else:
        distances = np.asarray(distances, dtype=float)
        if distances.shape != (n_classes, n_classes):
            raise ValueError("Distance matrix must be of shape (n_classes, n_classes).")

    diff_sq = (prev_real - prev_pred) ** 2
    total = 0.0
    for i in Y_star:
        for j in range(n_classes):
            total += distances[j, i] * diff_sq[j]

    denom = len(Y_star) * (n_classes - 1)
    rnod = np.sqrt(total / denom)
    return float(rnod)
