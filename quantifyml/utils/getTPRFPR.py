import numpy as np
import pandas as pd
from .get_scores import get_scores

def get_tprfpr(X, y, classifier, folds=10):
    
    scores = get_scores(X, y, classifier, folds)
    _class = np.unique(y)[1]
    
    unique_scores = np.linspace(0,1,101)
        
    TprFpr = pd.DataFrame(columns=['threshold','tpr', 'fpr'])
    total_positive = len(scores[scores[:, 1] == _class])
    total_negative = len(scores[scores[:, 1] != _class])  
    for threshold in unique_scores:
        fp = len(scores[(scores[:, 0] > threshold) & (scores[:, 1] != _class)])  
        tp = len(scores[(scores[:, 0] > threshold) & (scores[:, 1] == _class)])

        tpr = round(tp/total_positive,4) if total_positive != 0 else 0
        fpr = round(fp/total_negative,4) if total_negative != 0 else 0
    
        aux = pd.DataFrame([[threshold, tpr, fpr]])
        aux.columns = ['threshold', 'tpr', 'fpr']    
        TprFpr = pd.concat([None if TprFpr.empty else TprFpr, aux])

     
    return TprFpr.reset_index(drop=True)