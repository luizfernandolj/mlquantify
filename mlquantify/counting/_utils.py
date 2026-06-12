import numpy as np


def compute_table(y, y_pred, classes):
    r"""Compute the confusion matrix table for a binary classification task.

    Parameters
    ----------
    y : np.ndarray
        The true labels.
    y_pred : np.ndarray
        The predicted labels.
    classes : np.ndarray
        The unique classes in the dataset.

    Returns
    -------
    tuple
        A tuple containing the counts of True Positives, False Positives,
        False Negatives, and True Negatives respectively.
    """
    TP = np.logical_and(y == y_pred, y == classes[1]).sum()
    FP = np.logical_and(y != y_pred, y == classes[0]).sum()
    FN = np.logical_and(y != y_pred, y == classes[1]).sum()
    TN = np.logical_and(y == y_pred, y == classes[0]).sum()
    return TP, FP, FN, TN


def compute_tpr(TP, FN):
    r"""Compute the True Positive Rate (Recall) for a binary classification task.

    Parameters
    ----------
    TP : int
        The number of True Positives.
    FN : int
        The number of False Negatives.

    Returns
    -------
    float
        The True Positive Rate (Recall).
    """
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def compute_fpr(FP, TN):
    r"""Compute the False Positive Rate for a binary classification task.

    Parameters
    ----------
    FP : int
        The number of False Positives.
    TN : int
        The number of True Negatives.

    Returns
    -------
    float
        The False Positive Rate.
    """
    if FP + TN == 0:
        return 0
    return FP / (FP + TN)


def evaluate_thresholds (y, probabilities:np.ndarray, score_edges:str="fixed") -> tuple:
    r"""Evaluate a range of classification thresholds to compute the corresponding
    True Positive Rate (TPR) and False Positive Rate (FPR) for a binary quantification task.

    Parameters
    ----------
    y : np.ndarray
        The true labels.
    probabilities : np.ndarray
        The predicted probabilities (scores) for the positive class.
    classes : np.ndarray
        The unique classes in the dataset.

    Returns
    -------
    tuple
        A tuple of (thresholds, tprs, fprs), where:
        - thresholds is a numpy array of evaluated thresholds,
        - tprs is a numpy array of corresponding True Positive Rates,
        - fprs is a numpy array of corresponding False Positive Rates.
    """
    y = np.asarray(y)
    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.unique(y)

    if score_edges == "fixed":
        unique_scores = np.linspace(0, 1, 101)
    else:
        unique_scores = np.unique(probabilities)

    # Vectorised TPR/FPR sweep. For a threshold ``t`` a positive prediction means
    # ``score >= t``; the number of class-c scores at or above ``t`` is a survival
    # count read from the sorted scores in O((n + T) log n) instead of the original
    # O(T * n) Python loop over thresholds.
    pos_scores = np.sort(probabilities[y == classes[1]])
    neg_scores = np.sort(probabilities[y == classes[0]])
    n_pos = pos_scores.shape[0]
    n_neg = neg_scores.shape[0]

    # count(score >= t) == n - count(score < t) == n - searchsorted(., t, "left")
    tp = n_pos - np.searchsorted(pos_scores, unique_scores, side="left")
    fp = n_neg - np.searchsorted(neg_scores, unique_scores, side="left")

    tprs = tp / n_pos if n_pos > 0 else np.zeros_like(unique_scores, dtype=float)
    fprs = fp / n_neg if n_neg > 0 else np.zeros_like(unique_scores, dtype=float)

    return (unique_scores, np.asarray(tprs, dtype=float), np.asarray(fprs, dtype=float))