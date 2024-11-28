import numpy as np



def absolute_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
        
    abs_error = abs(prev_pred - prev_real).mean(axis=-1)
    
    return abs_error






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






def kullback_leibler_divergence(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    return prev_real * abs(np.log((prev_real / prev_pred)))





def squared_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))

    sq_abs_error = ((prev_pred - prev_real) ** 2).mean(axis=-1)
    
    return sq_abs_error








def mean_squared_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
        
    mean_sq_error = squared_error(prev_real, prev_pred).mean()
    
    return mean_sq_error







def normalized_absolute_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    
    abs_error = absolute_error(prev_real, prev_pred)
    
    z_abs_error = (2 * (1 - min(prev_real)))
    
    normalized = abs_error / z_abs_error
    
    return normalized






def normalized_kullback_leibler_divergence(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    
    euler = np.exp(kullback_leibler_divergence(prev_real, prev_pred))
    normalized = 2 * (euler / (euler + 1)) - 1
    
    return normalized






def relative_absolute_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))

    relative = (absolute_error(prev_real, prev_pred) / prev_real).mean(axis=-1)
    
    return relative








def normalized_relative_absolute_error(prev_real:np.any, prev_pred:np.any):
    if isinstance(prev_real, dict):
        prev_real = np.asarray(list(prev_real.values()))
    if isinstance(prev_pred, dict):
        prev_pred = np.asarray(list(prev_pred.values()))
    
    relative = relative_absolute_error(prev_real, prev_pred)
    
    z_relative = (len(prev_real) - 1 + ((1 - min(prev_real)) / min(prev_real))) / len(prev_real)
    
    normalized = relative/z_relative
    
    return normalized

