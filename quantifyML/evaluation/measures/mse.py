import numpy as np
from .se import squared_error

def mean_squared_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
        
    mean_sq_error = squared_error(prev_real, prev_pred).mean()
    
    return mean_sq_error