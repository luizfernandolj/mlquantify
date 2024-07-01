from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold


class Quantifier(ABC, BaseEstimator):
    
    @abstractmethod
    def fit(self, *args, **kwargs):
        ...
      
    @abstractmethod  
    def predict(self, *args, **kwargs):
        ...

class Utils(ABC):

    def is_binary(self, y):
        unique_values = np.unique(y)
        self.binary = set(unique_values).issubset({0, 1})
        return self.binary
    
    def one_vs_all(self, y):
        for label in np.unique(y):
            y_label = np.asarray([1 if _class == label else 0 for _class in y])
            yield label, y_label
    
    def get_values(self, X, y, clf, scores:bool=False, tprfpr:bool=False):
        values = {}
        
        if scores:
            values["scores"] = self.get_scores(X, y, clf)
        if tprfpr:
            values["tprfpr"] = self.get_tprfpr(X, y, clf)
            
        return values
    
    
    def get_scores(self, X, y, clf, folds=10):
        
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
    
        skf = StratifiedKFold(n_splits=folds)    
        scores = []
        labels = []
        
        for train_index,valid_index in skf.split(X,y):
            
            tr_data = pd.DataFrame(X.iloc[train_index])   #Train data and labels
            tr_lbl = y.iloc[train_index]
            
            valid_data = pd.DataFrame(X.iloc[valid_index])  #Validation data and labels
            valid_lbl = y.iloc[valid_index]
            
            clf.fit(tr_data, tr_lbl)
            
            scores.extend(clf.predict_proba(valid_data)[:,1])     #evaluating scores
            labels.extend(valid_lbl)
        
        scores = np.c_[scores,labels]
        
        return scores
    
    
    def get_tprfpr(self, X, y, clf):
        
        scores = self.get_scores(X, y, clf)
        
        unique_scores = np.linspace(0,1,101)
        
        TprFpr = pd.DataFrame(columns=['threshold','tpr', 'fpr'])
        total_positive = len(scores[scores[:, 1] == 1])
        total_negative = len(scores[scores[:, 1] == 0])  
        for threshold in unique_scores:
            fp = len(scores[(scores[:, 0] > threshold) & (scores[:, 1] == 0)])  
            tp = len(scores[(scores[:, 0] > threshold) & (scores[:, 1] == 1)])

            tpr = round(tp/total_positive,4) if total_positive != 0 else 0
            fpr = round(fp/total_negative,4) if total_negative != 0 else 0
        
            aux = pd.DataFrame([[threshold, tpr, fpr]])
            aux.columns = ['threshold', 'tpr', 'fpr']    
            TprFpr = pd.concat([None if TprFpr.empty else TprFpr, aux])

        
        return TprFpr.reset_index(drop=True)
        