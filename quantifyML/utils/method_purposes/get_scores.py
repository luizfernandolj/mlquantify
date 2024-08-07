import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_scores(X, y, learner, folds:int=10, learner_fitted:bool=False) -> tuple:
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
        
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