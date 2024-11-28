import numpy as np
from .kld import kullback_leibler_divergence

def normalized_kullback_leibler_divergence(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    
    euler = np.exp(kullback_leibler_divergence(prev_real, prev_pred))
    normalized = 2 * (euler / (euler + 1)) - 1
    
    return normalized