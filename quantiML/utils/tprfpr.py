import numpy as np


def compute_table(y, y_pred, classes):
    TP = np.logical_and(y == y_pred, y == classes[1]).sum()
    FP = np.logical_and(y != y_pred, y == classes[0]).sum()
    FN = np.logical_and(y != y_pred, y == classes[1]).sum()
    TN = np.logical_and(y == y_pred, y == classes[0]).sum()
    return TP, FP, FN, TN


def compute_tpr(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def compute_fpr(FP, TN):
    if FP + TN == 0:
        return 0
    return FP / (FP + TN)


def adjust_threshold(y, probabilities:np.ndarray, classes:np.ndarray) -> tuple:
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