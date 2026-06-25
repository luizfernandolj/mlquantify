import numpy as np

from ._utils import process_inputs


def AE(prev_real, prev_pred):
    r"""
    Compute the absolute error for each class or a dictionary of errors if input is a dictionary.

    Parameters
    ----------
    prev_real : array-like or dict
        True prevalence values for each class. If a dictionary, keys are class names, and values are prevalences.

    prev_pred : array-like or dict
        Predicted prevalence values for each class. If a dictionary, keys are class names, and values are prevalences.

    Returns
    -------
    error : array-like or dict
        Absolute error for each class. If input is a dictionary, returns a dictionary with errors for each class.
    """
    if isinstance(prev_real, dict):
        classes = prev_real.keys()
        prev_real, prev_pred = process_inputs(prev_real, prev_pred)
        abs_errors = np.abs(prev_pred - prev_real)
        return {class_: float(err) for class_, err in zip(classes, abs_errors)}
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return np.abs(prev_pred - prev_real)



def MAE(prev_real, prev_pred):
    r"""
    Compute the mean absolute error between the real and predicted prevalences.

    Parameters
    ----------
    prev_real : array-like of shape (n_classes,)
        True prevalence values for each class.

    prev_pred : array-like of shape (n_classes,)
        Predicted prevalence values for each class.

    Returns
    -------
    error : float
        Mean absolute error across all classes.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return np.mean(AE(prev_real, prev_pred))


def KLD(prev_real, prev_pred):
    r"""
    Compute the Kullback-Leibler divergence between the real and predicted prevalences.

    Parameters
    ----------
    prev_real : array-like of shape (n_classes,)
        True prevalence values for each class.

    prev_pred : array-like of shape (n_classes,)
        Predicted prevalence values for each class.

    Returns
    -------
    divergence : array-like of shape (n_classes,)
        Kullback-Leibler divergence for each class.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return prev_real * np.abs(np.log(prev_real / prev_pred))


def SE(prev_real, prev_pred):
    r"""
    Compute the mean squared error between the real and predicted prevalences.

    Parameters
    ----------
    prev_real : array-like of shape (n_classes,)
        True prevalence values for each class.

    prev_pred : array-like of shape (n_classes,)
        Predicted prevalence values for each class.

    Returns
    -------
    error : float
        Mean squared error across all classes.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return np.mean((prev_pred - prev_real) ** 2, axis=-1)


def MSE(prev_real, prev_pred):
    r""" Mean Squared Error

    Parameters
    ----------
    prev_real : array-like of shape (n_classes,)
        True prevalence values for each class.

    prev_pred : array-like of shape (n_classes,)
        Predicted prevalence values for each class.

    Returns
    -------
    mse : float
        Mean squared error across all classes.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return SE(prev_real, prev_pred).mean()


def NAE(prev_real, prev_pred):
    r"""
    Compute the normalized absolute error between the real and predicted prevalences.

    Parameters
    ----------
    prev_real : array-like of shape (n_classes,)
        True prevalence values for each class.

    prev_pred : array-like of shape (n_classes,)
        Predicted prevalence values for each class.

    Returns
    -------
    error : float
        Normalized absolute error across all classes.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    abs_error = MAE(prev_real, prev_pred)
    z_abs_error = 2 * (1 - np.min(prev_real))
    return abs_error / z_abs_error


def NKLD(prev_real, prev_pred):
    r"""
    Compute the normalized Kullback-Leibler divergence between the real and predicted prevalences.

    Parameters
    ----------
    prev_real : array-like of shape (n_classes,)
        True prevalence values for each class.

    prev_pred : array-like of shape (n_classes,)
        Predicted prevalence values for each class.

    Returns
    -------
    divergence : float
        Normalized Kullback-Leibler divergence across all classes.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    kl_divergence = KLD(prev_real, prev_pred)
    euler = np.exp(kl_divergence)
    return 2 * (euler / (euler + 1)) - 1


def RAE(prev_real, prev_pred):
    r"""
    Compute the relative absolute error between the real and predicted prevalences.

    Parameters
    ----------
    prev_real : array-like of shape (n_classes,)
        True prevalence values for each class.

    prev_pred : array-like of shape (n_classes,)
        Predicted prevalence values for each class.

    Returns
    -------
    error : float
        Relative absolute error across all classes.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return (MAE(prev_real, prev_pred) / prev_real).mean(axis=-1)


def NRAE(prev_real, prev_pred):
    r"""
    Compute the normalized relative absolute error between the real and predicted prevalences.

    Parameters
    ----------
    prev_real : array-like of shape (n_classes,)
        True prevalence values for each class.

    prev_pred : array-like of shape (n_classes,)
        Predicted prevalence values for each class.

    Returns
    -------
    error : float
        Normalized relative absolute error across all classes.
    """
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    relative = RAE(prev_real, prev_pred)
    z_relative = (len(prev_real) - 1 + ((1 - np.min(prev_real)) / np.min(prev_real))) / len(prev_real)
    return relative / z_relative

