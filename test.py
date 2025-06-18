from mlquantify.methods import PACC
from mlquantify.utils import get_real_prev
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

qtf = PACC(RandomForestClassifier())

qtf.fit(X_train, y_train)

pred = qtf.predict(X_test)

print(pred)
print(get_real_prev(y_test))