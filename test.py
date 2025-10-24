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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Carregar o dataset Iris
data = load_breast_cancer()
X, y = data.data, data.target

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Criar e treinar o modelo
rf = KNeighborsClassifier(n_neighbors=3)

rf.fit(X_train, y_train)
rf_train_pred = rf.predict_proba(X_train)
rf_pred = rf.predict_proba(X_test)

quantifier = PCC

# Usar o quantificador CC sem learner
quantifier1 = quantifier()
predictions1 = quantifier1.aggregate(rf_pred)
print("Predicted class prevalences 1:", predictions1)

# Usar o quantificador CC com learner
quantifier2 = quantifier(learner=rf)
quantifier2.fit(X_train, y_train)
predictions2 = quantifier2.predict(X_test)
print("Predicted class prevalences 2:", predictions2)

print("soma das prevalências 1:", np.sum(predictions1))
print("soma das prevalências 2:", np.sum(predictions2))