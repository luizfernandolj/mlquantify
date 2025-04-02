from mlquantify import set_arguments
from mlquantify.methods import EMQ
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, 100)
X_test = np.random.rand(10, 10)
y_test = np.random.randint(0, 2, 10)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# parameters
y_pred = model.predict(X_test) # predictions of the test set
posteriors_train = model.predict_proba(X_train) # predictions of the training set generated via cross validation
posteriors_test = model.predict_proba(X_test) # predictions of the test set
y_labels = y_train # Generated via cross validation
y_pred_train = model.predict(X_train) # predictions of the training set generated via cross validation

set_arguments(y_pred=y_pred,
              posteriors_train=posteriors_train,
              posteriors_test=posteriors_test,
              y_labels=y_labels,
              y_pred_train=y_pred_train)

quantifier = EMQ()
quantifier.fit(X_train, y_train)
pred = quantifier.predict(X_test)
print(pred)