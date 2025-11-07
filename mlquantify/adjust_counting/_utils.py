import numpy as np


def compute_table(y, y_pred, classes):
    """
    Compute the confusion matrix table for a binary classification task.
    
    
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
        A tuple containing the True Positives, False Positives, False Negatives, and True Negatives.
    """
    TP = np.logical_and(y == y_pred, y == classes[1]).sum()
    FP = np.logical_and(y != y_pred, y == classes[0]).sum()
    FN = np.logical_and(y != y_pred, y == classes[1]).sum()
    TN = np.logical_and(y == y_pred, y == classes[0]).sum()
    return TP, FP, FN, TN


def compute_tpr(TP, FN):
    """
    Compute the True Positive Rate (Recall) for a binary classification task.
    
    
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
    """
    Compute the False Positive Rate for a binary classification task.
    
    
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


def evaluate_thresholds (y, probabilities:np.ndarray, classes) -> tuple:
    """
    Adjust the threshold for a binary quantification task to maximize the True Positive Rate.
    
    
    Parameters
    ----------
    y : np.ndarray
        The true labels.
    probabilities : np.ndarray
        The predicted probabilities.
    classes : np.ndarray
        The unique classes in the dataset.
    
    
    Returns
    -------
    tuple
        The best True Positive Rate and False Positive Rate.
    """
    unique_scores = np.linspace(0, 1, 101)
    
    tprs = []
    fprs = []
    
    for threshold in unique_scores:
        y_pred = np.where(probabilities >= threshold, classes[1], classes[0])
        
        TP, FP, FN, TN = compute_table(y, y_pred, classes)
        
        tpr = compute_tpr(TP, FN)
        fpr = compute_fpr(FP, TN)
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    #best_tpr, best_fpr = self.adjust_threshold(np.asarray(tprs), np.asarray(fprs))
    return (unique_scores, np.asarray(tprs), np.asarray(fprs))