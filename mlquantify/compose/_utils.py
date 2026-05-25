import numpy as np


def class_representations_to_matrix(class_representations):
    """Convert per-class representation vectors into a column matrix.

    Stacks the class representation vectors as columns, producing the
    training mixing matrix ``M`` used in distribution-matching objectives.

    Parameters
    ----------
    class_representations : array-like of shape (n_classes, representation_dim)
        Per-class representation vectors returned by a fitted
        :class:`~mlquantify.representations.BaseRepresentation`.

    Returns
    -------
    M : ndarray of shape (representation_dim, n_classes)
        Training mixing matrix (transpose of the input).

    Raises
    ------
    ValueError
        If ``class_representations`` is not a 2-D array.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.compose._utils import class_representations_to_matrix
    >>> reps = np.array([[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])
    >>> M = class_representations_to_matrix(reps)
    >>> M.shape
    (3, 2)
    """
    class_representations = np.asarray(class_representations, dtype=float)

    if class_representations.ndim != 2:
        raise ValueError(
            "class_representations must have shape "
            "(n_classes, representation_dim)."
        )

    return class_representations.T


def validate_matrix_problem(M, q):
    """Validate and cast inputs for a linear mixing problem.

    Ensures that the mixing matrix ``M`` is 2-D, that the test
    representation ``q`` is 1-D, and that their shapes are compatible.

    Parameters
    ----------
    M : array-like of shape (representation_dim, n_classes)
        Training mixing matrix.
    q : array-like of shape (representation_dim,)
        Test representation vector.

    Returns
    -------
    M : ndarray of shape (representation_dim, n_classes)
        Validated and cast mixing matrix.
    q : ndarray of shape (representation_dim,)
        Validated and cast test representation.

    Raises
    ------
    ValueError
        If ``M`` is not 2-D, ``q`` is not 1-D, or the leading dimensions
        of ``M`` and ``q`` do not match.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.compose._utils import validate_matrix_problem
    >>> M = np.array([[0.3, 0.7], [0.6, 0.4]])
    >>> q = np.array([0.45, 0.55])
    >>> M_val, q_val = validate_matrix_problem(M, q)
    >>> M_val.shape, q_val.shape
    ((2, 2), (2,))
    """
    M = np.asarray(M, dtype=float)
    q = np.asarray(q, dtype=float)

    if M.ndim != 2:
        raise ValueError("M must be a 2D matrix.")

    if q.ndim != 1:
        raise ValueError("q must be a 1D vector.")

    if M.shape[0] != q.shape[0]:
        raise ValueError(
            f"Incompatible shapes: M has {M.shape[0]} rows, "
            f"but q has {q.shape[0]} elements."
        )

    return M, q