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
        print(prev_real)
        prev_real = np.asarray(prev_real)
    if isinstance(prev_pred, list):
        print(prev_pred)
        prev_pred = np.asarray(prev_pred)
    
    # Pad with zeros if lengths differ
    len_real = len(prev_real)
    len_pred = len(prev_pred)
    
    if len_real > len_pred:
        prev_pred = np.pad(prev_pred, (0, len_real - len_pred), constant_values=0)
    elif len_pred > len_real:
        prev_real = np.pad(prev_real, (0, len_pred - len_real), constant_values=0)
        
    return prev_real, prev_pred


def AE(prev_pred, prev_real):
    """
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
        prev_real, prev_pred = process_inputs(prev_pred, prev_real)
        abs_errors = np.abs(prev_pred - prev_real)
        return {class_: float(err) for class_, err in zip(classes, abs_errors)}
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    return np.abs(prev_pred - prev_real)



def MAE(prev_pred, prev_real):
    """
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
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    return np.mean(AE(prev_pred, prev_real))


def KLD(prev_pred, prev_real):
    """
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
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    return prev_real * np.abs(np.log(prev_real / prev_pred))


def SE(prev_pred, prev_real):
    """
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
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    return np.mean((prev_pred - prev_real) ** 2, axis=-1)


def MSE(prev_pred, prev_real):
    """ Mean Squared Error

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
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    return SE(prev_pred, prev_real).mean()


def NAE(prev_pred, prev_real):
    """
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
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    abs_error = MAE(prev_pred, prev_real)
    z_abs_error = 2 * (1 - np.min(prev_real))
    return abs_error / z_abs_error


def NKLD(prev_pred, prev_real):
    """
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
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    kl_divergence = KLD(prev_pred, prev_real)
    euler = np.exp(kl_divergence)
    return 2 * (euler / (euler + 1)) - 1


def RAE(prev_pred, prev_real):
    """
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
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    return (MAE(prev_pred, prev_real) / prev_real).mean(axis=-1)


def NRAE(prev_pred, prev_real):
    """
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
    prev_real, prev_pred = process_inputs(prev_pred, prev_real)
    relative = RAE(prev_pred, prev_real)
    z_relative = (len(prev_real) - 1 + ((1 - np.min(prev_real)) / np.min(prev_real))) / len(prev_real)
    return relative / z_relative

