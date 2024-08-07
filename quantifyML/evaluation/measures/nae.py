import numpy as np
from .ae import absolute_error

def normalized_absolute_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    
    abs_error = absolute_error(prev_real, prev_pred)
    
    z_abs_error = (2 * (1 - min(prev_real)))
    
    normalized = abs_error / z_abs_error
    
    return normalized