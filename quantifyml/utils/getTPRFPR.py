import pandas as pd
import numpy as np
from .get_scores import get_scores

def get_tprfpr(X, y, classifier, folds=10):
    scores = get_scores(X, y, classifier, folds)
    _class = np.unique(y)[1]

    unique_scores = np.linspace(0, 1, 101)

    total_positive = np.sum(scores[:, 1] == _class)
    total_negative = np.sum(scores[:, 1] != _class)
    
    TprFpr = pd.DataFrame({
        'threshold': unique_scores,
        'tpr': np.zeros(len(unique_scores)),
        'fpr': np.zeros(len(unique_scores))
    })

    for i, threshold in enumerate(unique_scores):
        tp = np.sum((scores[:, 0] > threshold) & (scores[:, 1] == _class))
        fp = np.sum((scores[:, 0] > threshold) & (scores[:, 1] != _class))

        TprFpr.loc[i, 'tpr'] = tp / total_positive if total_positive != 0 else 0
        TprFpr.loc[i, 'fpr'] = fp / total_negative if total_negative != 0 else 0

    TprFpr[['tpr', 'fpr']] = TprFpr[['tpr', 'fpr']]

    return TprFpr