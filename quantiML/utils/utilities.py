from . import *
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def parallel(func, classes, *args, **kwargs):
    return np.asarray(
        Parallel(n_jobs=-1, backend='threading')(
            delayed(func)(c, *args, **kwargs) for c in classes
        )
    )
    

def normalize_prevalence(prevalences: np.ndarray, classes:list):
    
    if isinstance(prevalences, dict):
        summ = sum(prevalences.values())
        prevalences = {_class:value/summ for _class, value in prevalences.items()}
        return prevalences
    
    summ = prevalences.sum(axis=-1, keepdims=True)
    prevalences = np.true_divide(prevalences, sum(prevalences), where=summ>0)
    prevalences = {_class:prev for _class, prev in zip(classes, prevalences)}
    
    return prevalences


def GetScores(X, y, learner, folds:int=10, learner_fitted:bool=False) -> tuple:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.DataFrame(y)
        
    if learner_fitted:
        probabilities = learner.predict_proba(X)
        y_label = y
    else:
    
        skf = StratifiedKFold(n_splits=folds)    
        probabilities = []
        y_label = []
        
        for train_index, valid_index in skf.split(X,y):
            
            tr_data = pd.DataFrame(X.iloc[train_index])   #Train data and labels
            tr_label = y.iloc[train_index]
            
            valid_data = pd.DataFrame(X.iloc[valid_index])  #Validation data and labels
            valid_label = y.iloc[valid_index]
            
            learner.fit(tr_data, tr_label)
            
            probabilities.extend(learner.predict_proba(valid_data))     #evaluating scores
            y_label.extend(valid_label)
    
    return np.asarray(y_label), np.asarray(probabilities)