import numpy as np

def normalize_prevalence(prevalences: np.ndarray, classes:list):
    
    if isinstance(prevalences, dict):
        summ = sum(prevalences.values())
        prevalences = {_class:value/summ for _class, value in prevalences.items()}
        return prevalences
    
    summ = prevalences.sum(axis=-1, keepdims=True)
    prevalences = np.true_divide(prevalences, sum(prevalences), where=summ>0)
    prevalences = {_class:prev for _class, prev in zip(classes, prevalences)}
    
    return prevalences