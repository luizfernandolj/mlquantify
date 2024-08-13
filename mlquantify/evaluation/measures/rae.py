import numpy as np
from . import absolute_error

def relative_absolute_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))

    relative = (absolute_error(prev_real, prev_pred) / prev_real).mean(axis=-1)
    
    return relative