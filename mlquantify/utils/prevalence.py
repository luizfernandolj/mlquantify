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
        summ = sum(prevalences.values())
        prevalences = {int(_class):float(value/summ) for _class, value in prevalences.items()}
        return prevalences
    
    summ = np.sum(prevalences, axis=-1, keepdims=True)
    prevalences = np.true_divide(prevalences, sum(prevalences), where=summ>0)
    prevalences = {int(_class):float(prev) for _class, prev in zip(classes, prevalences)}
    prevalences = defaultdict(lambda: 0, prevalences)
    
    # Ensure all classes are present in the result
    for cls in classes:
        prevalences[cls] = prevalences[cls]
    
    return dict(prevalences)

