"""Shared helpers for the :mod:`mlquantify.visualization` Display classes.

These utilities mirror the conventions of scikit-learn's ``*Display`` API
(lazy matplotlib import, ``ax``/``figure_`` handling, style-kwargs validation)
so that the quantification-specific plots feel familiar to anyone who has used
``RocCurveDisplay`` and friends.
"""

import numpy as np


def check_matplotlib_support(caller_name):
    """Raise a friendly error if matplotlib is not installed.

    Parameters
    ----------
    caller_name : str
        Name of the caller, used in the error message.
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:  # pragma: no cover - matplotlib is a hard dep
        raise ImportError(
            f"{caller_name} requires matplotlib. Install it with "
            "`pip install matplotlib`."
        ) from e


def _check_ax(ax):
    """Return ``(figure, ax)``, creating a new pair when ``ax`` is None."""
    check_matplotlib_support("mlquantify.visualization")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    return fig, ax


def _color_cycle():
    """Return the list of colors from the active matplotlib prop cycle."""
    import matplotlib.pyplot as plt

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color")
    return colors or [f"C{i}" for i in range(10)]


def _validate_style_kwargs(default_style_kwargs, user_style_kwargs):
    """Merge default and user matplotlib style kwargs, resolving aliases.

    User-supplied values win. Short aliases (``c``, ``ls``, ``lw`` ...) are
    normalised to their long form so a default and an alias never collide on
    the same matplotlib artist. Mirrors scikit-learn's helper of the same name.
    """
    invalid_to_valid_kw = {
        "ls": "linestyle",
        "c": "color",
        "ec": "edgecolor",
        "fc": "facecolor",
        "lw": "linewidth",
        "mec": "markeredgecolor",
        "mfc": "markerfacecolor",
        "mew": "markeredgewidth",
        "ms": "markersize",
    }
    for invalid_key, valid_key in invalid_to_valid_kw.items():
        if invalid_key in user_style_kwargs and valid_key in user_style_kwargs:
            raise TypeError(
                f"Got both {invalid_key!r} and {valid_key!r}, which are aliases "
                "of one another"
            )
    valid_kwargs = {**default_style_kwargs, **user_style_kwargs}
    for invalid_key, valid_key in invalid_to_valid_kw.items():
        if invalid_key in valid_kwargs:
            valid_kwargs[valid_key] = valid_kwargs.pop(invalid_key)
    return valid_kwargs


def _to_prevalence_array(prevalence, classes=None):
    """Coerce a prevalence estimate (dict, list, or array) to a 1-D float array.

    When ``prevalence`` is a dict and ``classes`` is given, the values are
    ordered to match ``classes``.
    """
    if isinstance(prevalence, dict):
        if classes is None:
            classes = list(prevalence.keys())
        return np.asarray([prevalence.get(c, 0.0) for c in classes], dtype=float)
    return np.asarray(prevalence, dtype=float)


def _default_class_names(class_names, n_classes):
    """Return a list of length ``n_classes`` of string class labels."""
    if class_names is None:
        return [str(i) for i in range(n_classes)]
    class_names = list(class_names)
    if len(class_names) != n_classes:
        raise ValueError(
            f"class_names has length {len(class_names)} but the prevalence "
            f"vectors have {n_classes} classes."
        )
    return [str(c) for c in class_names]
