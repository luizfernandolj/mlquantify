import numpy as np

def bias(prev_real:np.any, prev_pred:np.any):
    classes = None
    if isinstance(prev_real, dict):
        classes = prev_real.keys()
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    
    abs_errors = abs(prev_pred - prev_real)
    
    if classes:
        return {class_:float(abs_error) for class_, abs_error in zip(classes, abs_errors)}

    return abs_errors