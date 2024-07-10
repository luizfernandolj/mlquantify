from . import *
from joblib import Parallel, delayed
import numpy as np


def get_values(X, y, clf, scores:bool=False, tprfpr:bool=False):
        values = {}
        
        if scores:
            values["scores"] = get_scores(X, y, clf, 3)
        if tprfpr:
            values["tprfpr"] = get_tprfpr(X, y, clf, 3)
            
        return values


def parallel(func, classes, *args, **kwargs):
    return np.asarray(
        Parallel(n_jobs=-1, backend='threading')(
            delayed(func)(c, *args, **kwargs) for c in classes
        )
    )
    

def normalize_prevalence(prevalences: np.ndarray, classes:list):
    
    if isinstance(prevalences, dict):
        summ = sum(prevalences.values())
        print(summ)
        prevalences = {_class:value/summ for _class, value in prevalences}
        return prevalences
    
    summ = prevalences.sum(axis=-1, keepdims=True)
    prevalences = np.true_divide(prevalences, sum(prevalences), where=summ>0)
    prevalences = {_class:prev for _class, prev in zip(classes, prevalences)}
    
    return prevalences