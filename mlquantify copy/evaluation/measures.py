import numpy as np

def process_inputs(prev_real, prev_pred):
    """
    .. :noindex:
    
    Process the input data for internal use.
    """
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    return prev_real, prev_pred


def absolute_error(prev_real, prev_pred):
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
        prev_real, prev_pred = process_inputs(prev_real, prev_pred)
        abs_errors = np.abs(prev_pred - prev_real)
        return {class_: float(err) for class_, err in zip(classes, abs_errors)}
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return np.abs(prev_pred - prev_real)



def mean_absolute_error(prev_real, prev_pred):
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
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return np.mean(absolute_error(prev_real, prev_pred))


def kullback_leibler_divergence(prev_real, prev_pred):
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
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return prev_real * np.abs(np.log(prev_real / prev_pred))


def squared_error(prev_real, prev_pred):
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
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return np.mean((prev_pred - prev_real) ** 2, axis=-1)


def mean_squared_error(prev_real, prev_pred):
    """
    Compute the mean squared error across all classes.

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
    return squared_error(prev_real, prev_pred).mean()


def normalized_absolute_error(prev_real, prev_pred):
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
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    abs_error = mean_absolute_error(prev_real, prev_pred)
    z_abs_error = 2 * (1 - np.min(prev_real))
    return abs_error / z_abs_error


def normalized_kullback_leibler_divergence(prev_real, prev_pred):
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
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    kl_divergence = kullback_leibler_divergence(prev_real, prev_pred)
    euler = np.exp(kl_divergence)
    return 2 * (euler / (euler + 1)) - 1


def relative_absolute_error(prev_real, prev_pred):
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
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    return (mean_absolute_error(prev_real, prev_pred) / prev_real).mean(axis=-1)


def normalized_relative_absolute_error(prev_real, prev_pred):
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
    prev_real, prev_pred = process_inputs(prev_real, prev_pred)
    relative = relative_absolute_error(prev_real, prev_pred)
    z_relative = (len(prev_real) - 1 + ((1 - np.min(prev_real)) / np.min(prev_real))) / len(prev_real)
    return relative / z_relative

