import numpy as np
from collections import defaultdict

def normalize_prevalence(prevalences: np.ndarray, classes:list):
    
    if isinstance(prevalences, dict):
        summ = sum(prevalences.values())
        prevalences = {int(_class):float(value/summ) for _class, value in prevalences.items()}
        return prevalences
    
    summ = np.sum(prevalences, axis=-1, keepdims=True)
    prevalences = np.true_divide(prevalences, sum(prevalences), where=summ>0)
    prevalences = {int(_class):float(prev) for _class, prev in zip(classes, prevalences)}
    prevalences = defaultdict(lambda: 0, prevalences)
    
    # Ensure all classes are present in the result
    for cls in classes:
        prevalences[cls] = prevalences[cls]
    
    return dict(prevalences)