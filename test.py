from mlquantify.methods.aggregative import EMQ
from sklearn.linear_model import LogisticRegression
import numpy as np

X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=100)
X_test = np.random.rand(50, 10)

emq = EMQ(LogisticRegression())
emq.fit(X_train, y_train)

class_distribution = emq.predict(X_test)
scores = emq.predict_proba(X_test)

print("Class distribution:", class_distribution)
print("Scores:", scores)