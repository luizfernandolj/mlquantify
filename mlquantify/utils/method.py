import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold



def sqEuclidean(dist1, dist2):
    """
    Compute the squared Euclidean distance between two probability distributions.

    The squared Euclidean distance is a measure of dissimilarity between two probability
    distributions. It is defined as:

        D(P, Q) = Σ(Pᵢ - Qᵢ)²

    Parameters
    ----------
    dist1 : array-like
        The first probability distribution \( P \), where each element \( Pᵢ \) represents
        the probability of the \( i \)-th event.
    dist2 : array-like
        The second probability distribution \( Q \), where each element \( Qᵢ \) represents
        the probability of the \( i \)-th event.

    Returns
    -------
    float
        The squared Euclidean distance between the two distributions.

    Notes
    -----
    - This distance is non-negative and equals zero if and only if the two distributions
      are identical.
    - Both input distributions must be valid probability distributions; their elements
      should be non-negative and sum to 1.
    """
    P = dist1
    Q = dist2
    return sum((P - Q)**2)

    
def probsymm(dist1, dist2):
    """
    Compute the probabilistic symmetric distance between two probability distributions.

    The probabilistic symmetric distance is a measure of dissimilarity between two probability
    distributions. It is defined as:

        D(P, Q) = 2 * Σ((Pᵢ - Qᵢ)² / (Pᵢ + Qᵢ))

    Parameters
    ----------
    dist1 : array-like
        The first probability distribution \( P \), where each element \( Pᵢ \) represents
        the probability of the \( i \)-th event.
    dist2 : array-like
        The second probability distribution \( Q \), where each element \( Qᵢ \) represents
        the probability of the \( i \)-th event.

    Returns
    -------
    float
        The probabilistic symmetric distance between the two distributions.

    Notes
    -----
    - This distance is non-negative and equals zero if and only if the two distributions
      are identical.
    - Both input distributions must be valid probability distributions; their elements
      should be non-negative and sum to 1.
    - Division by zero is avoided by assuming the input distributions have no zero elements.
    """
    P = dist1
    Q = dist2
    return 2 * sum((P - Q)**2 / (P + Q))


def topsoe(dist1, dist2):
    """
    Compute the Topsøe distance between two probability distributions.

    The Topsøe distance is a measure of divergence between two probability distributions.
    It is defined as:

        D(P, Q) = Σ(Pᵢ * log(2 * Pᵢ / (Pᵢ + Qᵢ)) + Qᵢ * log(2 * Qᵢ / (Pᵢ + Qᵢ)))

    Parameters
    ----------
    dist1 : array-like
        The first probability distribution \( P \), where each element \( Pᵢ \) represents
        the probability of the \( i \)-th event.
    dist2 : array-like
        The second probability distribution \( Q \), where each element \( Qᵢ \) represents
        the probability of the \( i \)-th event.

    Returns
    -------
    float
        The Topsøe distance between the two distributions.

    Notes
    -----
    - This distance is non-negative and equals zero if and only if the two distributions
      are identical.
    - Both input distributions must be valid probability distributions; their elements
      should be non-negative and sum to 1.
    - Division by zero is avoided by assuming the input distributions have no zero elements.
    - The logarithm used is the natural logarithm.
    """
    P = dist1
    Q = dist2
    return sum(P * np.log(2 * P / (P + Q)) + Q * np.log(2 * Q / (P + Q)))


def hellinger(dist1, dist2):
    """
    Compute the Hellinger distance between two probability distributions.

    The Hellinger distance is a measure of similarity between two probability distributions.
    It is defined as:

        H(P, Q) = 2 * sqrt(|1 - Σ√(Pᵢ * Qᵢ)|)

    Parameters
    ----------
    dist1 : array-like
        The first probability distribution \( P \), where each element \( Pᵢ \) represents
        the probability of the \( i \)-th event.
    dist2 : array-like
        The second probability distribution \( Q \), where each element \( Qᵢ \) represents
        the probability of the \( i \)-th event.

    Returns
    -------
    float
        The Hellinger distance between the two distributions.

    Notes
    -----
    - The Hellinger distance ranges from 0 to 2, where 0 indicates that the distributions
      are identical, and 2 indicates that they are completely different.
    - Both input distributions must be valid probability distributions; their elements
      should be non-negative and sum to 1.
    - The absolute value is used to handle numerical errors that may cause the expression
      inside the square root to become slightly negative.
    """
    P=dist1
    Q=dist2
    return 2 * np.sqrt(np.abs(1 - sum(np.sqrt(P * Q))))













def get_scores(X, y, learner, folds: int = 10, learner_fitted: bool = False) -> tuple:
    """
    Generate true labels and predicted probabilities using a machine learning model.

    This function evaluates a machine learning model using cross-validation or directly 
    with a pre-fitted model, returning the true labels and predicted probabilities.

    Parameters
    ----------
    X : Union[np.ndarray, pd.DataFrame]
        Input features for the model.
    y : Union[np.ndarray, pd.Series]
        Target labels corresponding to the input features.
    learner : object
        A machine learning model that implements the `fit` and `predict_proba` methods.
    folds : int, optional
        Number of folds for stratified cross-validation. Defaults to 10.
    learner_fitted : bool, optional
        If `True`, assumes the learner is already fitted and directly predicts probabilities 
        without performing cross-validation. Defaults to `False`.

    Returns
    -------
    tuple
        - An array of true labels.
        - An array of predicted probabilities.

    Notes
    -----
    - When `learner_fitted` is `True`, the model is assumed to be pre-trained and no 
      cross-validation is performed.
    - When `learner_fitted` is `False`, stratified k-fold cross-validation is used to 
      generate predictions.
    - The input data `X` and `y` are converted to pandas objects for compatibility.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    if learner_fitted:
        probabilities = learner.predict_proba(X)
        y_label = y
    else:
        skf = StratifiedKFold(n_splits=folds)
        probabilities = []
        y_label = []

        for train_index, valid_index in skf.split(X, y):
            tr_data = pd.DataFrame(X.iloc[train_index])  # Train data and labels
            tr_label = y.iloc[train_index]
            valid_data = pd.DataFrame(X.iloc[valid_index])  # Validation data and labels
            valid_label = y.iloc[valid_index]

            learner.fit(tr_data, tr_label)
            probabilities.extend(learner.predict_proba(valid_data))  # Evaluating scores
            y_label.extend(valid_label)

    return np.asarray(y_label), np.asarray(probabilities)






def getHist(scores, nbins):
    """
    Calculate histogram-like bin probabilities for a given set of scores.

    This function divides the score range into equal bins and computes the proportion 
    of scores in each bin, normalized by the total count.

    Parameters
    ----------
    scores : np.ndarray
        A 1-dimensional array of scores.
    nbins : int
        Number of bins for dividing the score range.

    Returns
    -------
    np.ndarray
        An array containing the normalized bin probabilities.

    Notes
    -----
    - The bins are equally spaced between 0 and 1, with an additional upper boundary 
      to include the maximum score.
    - The returned probabilities are normalized to account for the total number of scores.
    """
    breaks = np.linspace(0, 1, int(nbins) + 1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks, 1.1)

    re = np.repeat(1 / (len(breaks) - 1), (len(breaks) - 1))
    for i in range(1, len(breaks)):
        re[i - 1] = (re[i - 1] + len(np.where((scores >= breaks[i - 1]) & (scores < breaks[i]))[0])) / (len(scores) + 1)

    return re







def MoSS(n:int, alpha:float, m:float):
    """
    Generate a synthetic dataset using the MoSS method.

    Parameters
    ----------
    n : int
        The number of samples to generate.
    alpha : float
        The proportion of positive samples in the dataset.
    m : float
        The shape parameter for the synthetic dataset.

    Returns
    -------
    tuple
        A tuple containing the synthetic positive and negative samples.
    """
    n_pos = int(n*alpha)
    n_neg = int((1-alpha)*n)

    x_pos = np.arange(1, n_pos, 1)
    x_neg = np.arange(1, n_neg, 1)
    
    syn_plus = np.power(x_pos/(n_pos+1), m)
    syn_neg = 1 - np.power(x_neg/(n_neg+1), m)

    #moss = np.union1d(syn_plus, syn_neg)

    return syn_plus, syn_neg





def ternary_search(left, right, f, eps=1e-4):
    """This function applies Ternary search
    
    Parameters
    ----------
    left : float
        The left boundary of the search interval.
    right : float
        The right boundary of the search interval.
    f : function
        The function to optimize.
    eps : float, optional
        The desired precision of the result. Defaults to 1e-4.
    
    Returns
    -------
    float
        The value of the argument that minimizes the function.
    """

    while True:
        if abs(left - right) < eps:
            return(left + right) / 2
    
        leftThird  = left + (right - left) / 3
        rightThird = right - (right - left) / 3
    
        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird 
            
            
            
        








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


def adjust_threshold(y, probabilities:np.ndarray, classes:np.ndarray) -> tuple:
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