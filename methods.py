from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd



from mlquantify.adjust_counting import (
    PCC, 
    CC, 
    FM, 
    GAC,
    GPAC,
    ACC,
    X_method,
    MAX,
    T50,
    MS,
    MS2,
)
from mlquantify.likelihood import (
    EMQ,
    MLPE,
    CDE
)
from mlquantify.mixture import (
    DyS,
    HDy,
    SMM,
    SORD,
    HDx
)
from mlquantify.neighbors import (
    KDEyCS,
    KDEyHD,
    KDEyML,
    PWK
)

from mlquantify.utils import get_prev_from_labels



METHODS = {
    'CC': CC,
    'PCC': PCC,
    'FM': FM,
    'GAC': GAC,
    'GPAC': GPAC,
    'ACC': ACC,
    'X_method': X_method,
    'MAX': MAX,
    'T50': T50,
    'MS': MS,
    'MS2': MS2,
    'EMQ': EMQ,
    'MLPE': MLPE,
    'CDE': CDE,
    'DyS': DyS,
    'HDy': HDy,
    'SMM': SMM,
    'SORD': SORD,
    'HDx': HDx,
    'KDEyCS': KDEyCS,
    'KDEyHD': KDEyHD,
    'KDEyML': KDEyML,
    'PWK': PWK,
}



def load_data(name='iris', format='pandas', class_format='categorical'):
    if name == 'iris':
        data = load_iris()
    elif name == 'breast_cancer':
        data = load_breast_cancer()
    else:
        raise ValueError("Dataset not recognized. Use 'iris' or 'breast_cancer'.")
    
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    


    if format == 'pandas':
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
        y_train = pd.Series(y_train)
        y_test = pd.Series(y_test)
        


    if class_format == 'categorical' and name == 'breast_cancer':
        if isinstance(y_train, pd.Series):
            y_train = y_train.replace({0: 'malignant', 1: 'benign'})
        else:
            y_train = np.where(y_train == 0, 'malignant', np.where(y_train == 1, 'benign', y_train))

        if isinstance(y_test, pd.Series):
            y_test = y_test.replace({0: 'malignant', 1: 'benign'})
        else:
            y_test = np.where(y_test == 0, 'malignant', np.where(y_test == 1, 'benign', y_test))
    elif class_format == 'categorical' and name == 'iris':
        if isinstance(y_train, pd.Series):
            y_train = y_train.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        else:
            y_train = np.where(y_train == 0, 'setosa', np.where(y_train == 1, 'versicolor', np.where(y_train == 2, 'virginica', y_train)))

        if isinstance(y_test, pd.Series):
            y_test = y_test.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        else:
            y_test = np.where(y_test == 0, 'setosa', np.where(y_test == 1, 'versicolor', np.where(y_test == 2, 'virginica', y_test)))

    return X_train, X_test, y_train, y_test




def get_learner(learner_name):
    if learner_name == 'RandomForest':
        return RandomForestClassifier()
    elif learner_name == 'DecisionTree':
        return DecisionTreeClassifier()
    else:
        raise ValueError("Learner not recognized. Use 'RandomForest' or 'DecisionTree'.")
    
    
    
    
def get_predictions(X_train, y_train, X_test, learner, type_predictions="proba"):

    learner.fit(X_train, y_train)

    if type_predictions == "proba":
        train_predictions = learner.predict_proba(X_train)
        test_predictions = learner.predict_proba(X_test)
    elif type_predictions == "labels":
        train_predictions = learner.predict(X_train)
        test_predictions = learner.predict(X_test)

    return test_predictions, train_predictions, y_train