from joblib import Parallel, delayed
import numpy as np


def parallel(func, elements, n_jobs: int = 1, *args):
    return Parallel(n_jobs=n_jobs)(
        delayed(func)(e, *args) for e in elements
    )

    