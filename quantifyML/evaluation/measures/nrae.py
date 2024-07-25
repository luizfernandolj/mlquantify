import numpy as np
from .rae import relative_absolute_error

def normalized_relative_absolute_error(prev_real:np.any, prev_pred:np.any):
    
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    
    relative = relative_absolute_error(prev_real, prev_pred)
    
    z_relative = (len(prev_real) - 1 + ((1 - min(prev_real)) / min(prev_real))) / len(prev_real)
    
    normalized = relative/z_relative
    
    return normalized