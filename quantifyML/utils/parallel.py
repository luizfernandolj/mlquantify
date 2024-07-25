from joblib import Parallel, delayed
import numpy as np


def parallel(func, classes, *args, **kwargs):
    return np.asarray(
        Parallel(n_jobs=-1, backend='threading')(
            delayed(func)(c, *args, **kwargs) for c in classes
        )
    )
    