import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_scores(X_train, Y_train, clf, folds=10):
    
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(Y_train, np.ndarray):
        Y_train = pd.DataFrame(Y_train)
    
    skf = StratifiedKFold(n_splits=folds)    
    results = []
    class_labl = []
    
    for train_index, valid_index in skf.split(X_train,Y_train):
        
        tr_data = pd.DataFrame(X_train[train_index])   #Train data and labels
        tr_lbl = Y_train[train_index]
        
        valid_data = pd.DataFrame(X_train[valid_index])  #Validation data and labels
        valid_lbl = Y_train[valid_index]
        
        clf.fit(tr_data, tr_lbl)
        
        results.extend(clf.predict_proba(valid_data)[:,1])     #evaluating scores
        class_labl.extend(valid_lbl)
    
    scores = np.c_[results,class_labl]
    
    return scores