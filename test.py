from mlquantify.adjust_counting import PCC, CC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# Carregar o dataset Iris
data = load_iris()
X, y = data.data, data.target

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Criar e treinar o modelo
dt = KNeighborsClassifier(n_neighbors=3)

dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

#dt_pred = dt_pred[:, 1]

# Usar o quantificador CC sem learner
cc1 = CC()
predictions1 = cc1.aggregate(dt_pred)
print("Predicted class prevalences 1:", predictions1)

# Usar o quantificador CC com learner
cc2 = CC(learner=dt)
cc2.fit(X_train, y_train)
predictions2 = cc2.predict(X_test)
print("Predicted class prevalences 2:", predictions2)

print("soma das prevalências 1:", np.sum(predictions1))
print("soma das prevalências 2:", np.sum(predictions2))

