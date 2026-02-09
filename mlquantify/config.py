"""Global configuration state and functions for management"""
import threading
from contextlib import contextmanager as contextmanager


_global_config = {
    "prevalence_return_type": "dict",
    "prevalence_normalization": "mean",
}
_threadlocal = threading.local()


def _get_threadlocal_config():
    r"""Get a threadlocal **mutable** configuration. If the configuration
    does not exist, copy the default global configuration."""
    if not hasattr(_threadlocal, "global_config"):
        _threadlocal.global_config = _global_config.copy()
    return _threadlocal.global_config


def get_config():
    r"""Retrieve the current mlquantify configuration.

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global mlquantify configuration.
    set_config : Set global mlquantify configuration.

    Examples
    --------
    >>> import mlquantify
    >>> config = mlquantify.get_config()
    >>> config.keys()
    dict_keys(['prevalence_return_type', 'prevalence_normalization'])
    """
    # Return a copy of the threadlocal configuration so that users will
    # not be able to modify the configuration with the returned dict.
    return _get_threadlocal_config().copy()


def set_config(
    prevalence_return_type=None,
    prevalence_normalization=None,
):
    r"""Set global mlquantify configuration.

    Parameters
    ----------
    prevalence_return_type : {'dict', 'array'}, default=None
        The format of the returned prevalence estimates:
        - 'dict': Returns a dictionary mapping class labels to values.
        - 'array': Returns a numpy array of values.
        Global default: 'dict'.

    prevalence_normalization : {'sum', 'l1', 'softmax', 'mean', 'median', None}, default=None
        The strategy for normalizing or aggregating prevalence estimates:
        - 'sum' or 'l1': Normalizes values so that they sum to 1.
        - 'softmax': Applies the softmax function to the estimates.
        - 'mean': Takes the arithmetic mean of multiple estimates.
        - 'median': Takes the median of multiple estimates.
        - None: No normalization or aggregation is performed.
        Global default: 'mean'.

    See Also
    --------
    config_context : Context manager for global mlquantify configuration.
    get_config : Retrieve current values of the global configuration.

    Examples
    --------
    >>> from mlquantify import set_config, get_config
    >>> set_config(prevalence_return_type='array', prevalence_normalization='probs')
    >>> get_config()['prevalence_normalization']
    'probs'
    """
    local_config = _get_threadlocal_config()

    if prevalence_return_type is not None:
        local_config["prevalence_return_type"] = prevalence_return_type
    if prevalence_normalization is not None:
        local_config["prevalence_normalization"] = prevalence_normalization


@contextmanager
def config_context(
    *,
    prevalence_return_type=None,
    prevalence_normalization=None,
):
    r"""Context manager to temporarily change the global mlquantify configuration.

    Parameters
    ----------
    prevalence_return_type : {'dict', 'array'}, default=None
        If 'dict', validate_prevalences returns a dictionary.
        If 'array', validate_prevalences returns a numpy array.
        If None, the existing configuration won't change.
        Global default: 'dict'.

    prevalence_normalization : {'sum', 'l1', 'softmax', 'mean', 'median', None}, default=None
        Default normalization or aggregation strategy for validate_prevalences.
        If None, the existing configuration won't change.
        Global default: 'mean'.

    Yields
    ------
    None.

    See Also
    --------
    set_config : Set global mlquantify configuration.
    get_config : Retrieve current values of the global configuration.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    Examples
    --------
    >>> import mlquantify
    >>> from mlquantify import config_context
    >>> with config_context(prevalence_return_type='array'):
    ...     mlquantify.get_config()['prevalence_return_type']
    'array'
    >>> mlquantify.get_config()['prevalence_return_type']
    'dict'
    """
    old_config = get_config()
    set_config(
        prevalence_return_type=prevalence_return_type,
        prevalence_normalization=prevalence_normalization,
    )

    try:
        yield
    finally:
        set_config(**old_config)
