import numpy as np
from .ae import absolute_error

def squared_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))

    sq_abs_error = ((prev_pred - prev_real) ** 2).mean(axis=-1)
    
    return sq_abs_error