from mlquantify.evaluation.protocol import APP, UPP, PPP, NPP
from mlquantify.evaluation.measures import mean_absolute_error
from mlquantify.methods import DyS
from mlquantify.utils import get_real_prev
from sklearn.ensemble import RandomForestClassifier
import mlquantify as mq
import numpy as np

X_train = np.random.rand(1000, 20)  # training data
y_train = np.random.randint(0, 2, size=1000)  # training labels
X_test = np.random.rand(1000, 20)  # test data
y_test = np.random.randint(0, 2, size=1000)  # test labels


classificador = None

dys = DyS(classificador)

if classificador is None:
    dys.fit(scores, classe)
else:
    dys.fit(X_train, y)
    

protocol = PPP(batch_size=200, prevalences=[0.1, 0.5, 0.2, 0.3])

print(len(list(protocol.split(X_test, y_test))))

for idx in protocol.split(X_test, y_test):
    X_batch = X_test[idx]
    y_batch = y_test[idx]
    
    y_pred = dys.predict(scores=scores)

    y_real = get_real_prev(y_batch)

    print(y_real)