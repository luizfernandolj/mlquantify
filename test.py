import numpy as np
import pandas as pd
import time
from quapy.data import LabelledCollection
from quapy.method.aggregative import DyS
from quapy.method.meta import Ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from quapy.error import ae

# Leitura do dataset
def read_dataset(file_path, label_column):
    df = pd.read_csv(file_path)
    X = df.drop(columns=[label_column]).values
    y = df[label_column].values
    return X, y

# Função para criar um LabelledCollection a partir de arrays X e y
def create_labelled_collection(X, y):
    return LabelledCollection(X, y)

# Função principal para rodar o ensemble
def run_ensemble(file_path, label_column, clf, ensemble_size=50, test_size=0.3):
    X, y = read_dataset(file_path, label_column)
    
    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    train_collection = create_labelled_collection(X_train, y_train)
    test_collection = create_labelled_collection(X_test, y_test)

    # Criação do ensemble
    base_learner = MS(clf)  # Classificação de Conteúdo
    ensemble = Ensemble(base_learner, size=ensemble_size, policy="ptr", verbose=True, n_jobs=-1)
    
    # Treinamento do ensemble
    ensemble.fit(train_collection)
    
    # Avaliação do ensemble
    predictions = ensemble.quantify(test_collection.instances)
    
    # Retornar as previsões e a acurácia
    accuracy = (test_collection.prevalence(), predictions)
    
    return predictions, accuracy

# Exemplo de uso
file_path = 'data/click-prediction.csv'
label_column = 'class'
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

start = time.time()
predictions, accuracy = run_ensemble(file_path, label_column, clf)
end = time.time()

total_time = end-start

print(f"time: {total_time} seconds")

print(f'Accuracy: {accuracy}')

