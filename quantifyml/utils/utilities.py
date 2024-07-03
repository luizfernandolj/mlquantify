from . import *
from joblib import Parallel, delayed
import numpy as np


def get_values(X, y, clf, scores:bool=False, tprfpr:bool=False):
        values = {}
        
        if scores:
            values["scores"] = get_scores(X, y, clf)
        if tprfpr:
            values["tprfpr"] = get_tprfpr(X, y, clf)
            
        return values


def parallel(func, classes, *args, **kwargs):
    return np.asarray(
        Parallel(n_jobs=-1, backend='threading')(
            delayed(func)(c, *args, **kwargs) for c in classes
        )
    )