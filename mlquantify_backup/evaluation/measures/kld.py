import numpy as np

def kullback_leibler_divergence(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    return prev_real * abs(np.log((prev_real / prev_pred)))