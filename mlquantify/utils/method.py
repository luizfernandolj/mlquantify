import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold



def sqEuclidean(dist1, dist2):
    P=dist1 
    Q=dist2 
    return sum((P-Q)**2)
    
def probsymm(dist1, dist2):
    P=dist1
    Q=dist2
    return 2*sum((P-Q)**2/(P+Q))

def topsoe(dist1, dist2):
    P=dist1
    Q=dist2
    return sum(P*np.log(2*P/(P+Q))+Q*np.log(2*Q/(P+Q)))

def hellinger(dist1, dist2):
    P=dist1
    Q=dist2
    return 2 * np.sqrt(np.abs(1 - sum(np.sqrt(P * Q))))













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





def getHist(scores, nbins):

    breaks = np.linspace(0, 1, int(nbins)+1)
    breaks = np.delete(breaks, -1)
    breaks = np.append(breaks,1.1)
    
    re = np.repeat(1/(len(breaks)-1), (len(breaks)-1))  
    for i in range(1,len(breaks)):
        re[i-1] = (re[i-1] + len(np.where((scores >= breaks[i-1]) & (scores < breaks[i]))[0]) ) / (len(scores)+ 1)

    return re






def MoSS(n:int, alpha:float, m:float):
  
  n_pos = int(n*alpha)
  n_neg = int((1-alpha)*n)

  x_pos = np.arange(1, n_pos, 1)
  x_neg = np.arange(1, n_neg, 1)
  
  syn_plus = np.power(x_pos/(n_pos+1), m)
  syn_neg = 1 - np.power(x_neg/(n_neg+1), m)

  #moss = np.union1d(syn_plus, syn_neg)

  return syn_plus, syn_neg





def ternary_search(left, right, f, eps=1e-4):
    """This function applies Ternary search"""

    while True:
        if abs(left - right) < eps:
            return(left + right) / 2
    
        leftThird  = left + (right - left) / 3
        rightThird = right - (right - left) / 3
    
        if f(leftThird) > f(rightThird):
            left = leftThird
        else:
            right = rightThird 
            
            
            
        








def compute_table(y, y_pred, classes):
    TP = np.logical_and(y == y_pred, y == classes[1]).sum()
    FP = np.logical_and(y != y_pred, y == classes[0]).sum()
    FN = np.logical_and(y != y_pred, y == classes[1]).sum()
    TN = np.logical_and(y == y_pred, y == classes[0]).sum()
    return TP, FP, FN, TN


def compute_tpr(TP, FN):
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)


def compute_fpr(FP, TN):
    if FP + TN == 0:
        return 0
    return FP / (FP + TN)


def adjust_threshold(y, probabilities:np.ndarray, classes:np.ndarray) -> tuple:
    unique_scores = np.linspace(0, 1, 101)
    
    tprs = []
    fprs = []
    
    for threshold in unique_scores:
        y_pred = np.where(probabilities >= threshold, classes[1], classes[0])
        
        TP, FP, FN, TN = compute_table(y, y_pred, classes)
        
        tpr = compute_tpr(TP, FN)
        fpr = compute_fpr(FP, TN)
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    #best_tpr, best_fpr = self.adjust_threshold(np.asarray(tprs), np.asarray(fprs))
    return (unique_scores, np.asarray(tprs), np.asarray(fprs))