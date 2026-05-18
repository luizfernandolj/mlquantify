import numpy as np


def class_representations_to_matrix(class_representations):
    class_representations = np.asarray(class_representations, dtype=float)

    if class_representations.ndim != 2:
        raise ValueError(
            "class_representations must have shape "
            "(n_classes, representation_dim)."
        )

    return class_representations.T


def validate_matrix_problem(M, q):
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