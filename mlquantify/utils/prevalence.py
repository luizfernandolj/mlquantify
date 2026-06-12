import numpy as np
import pandas as pd
from collections import defaultdict


def get_prev_from_labels(y, format="dict", classes: list = None):
    """
    Get the real prevalence of each class in the target array.

    Parameters
    ----------
    y : np.ndarray or pd.Series
        Array of class labels.
    format : str, default="dict"
        Format of the output. Can be "array" or "dict".
    classes : list, optional
        List of unique classes. If provided, the output will be sorted by these classes.

    Returns
    -------
    dict or np.ndarray
        Dictionary of class labels and their corresponding prevalence or array of prevalences.
    """
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    counts = y.value_counts(normalize=True).sort_index()

    if classes is not None:
        counts = counts.reindex(classes, fill_value=0.0)

    if format == "array":
        return counts.values

    real_prevs = counts.to_dict()
    return real_prevs





def normalize_prevalence(prevalences: np.ndarray, classes:list):
    """
    Normalize the prevalence of each class to sum to 1.
    
    Parameters
    ----------
    prevalences : np.ndarray
        Array of prevalences.
    classes : list
        List of unique classes.
    
    Returns
    -------
    dict
        Dictionary of class labels and their corresponding prevalence.
    """
    if isinstance(prevalences, dict):
        total = sum(prevalences.values())
        if total > 0:
            return {int(_class): float(value / total)
                    for _class, value in prevalences.items()}
        n = len(prevalences)
        return {int(_class): 1.0 / n for _class in prevalences}

    prevalences = np.asarray(prevalences, dtype=float)
    total = prevalences.sum()
    if total > 0:
        prevalences = prevalences / total
    else:
        # degenerate (non-positive sum): fall back to a uniform distribution
        # rather than dividing by zero / returning uninitialised values.
        prevalences = np.full(prevalences.shape[-1], 1.0 / prevalences.shape[-1])

    prevalences = {int(_class): float(prev) for _class, prev in zip(classes, prevalences)}
    prevalences = defaultdict(lambda: 0, prevalences)

    # Ensure all classes are present in the result
    for cls in classes:
        prevalences[cls] = prevalences[cls]

    return dict(prevalences)

