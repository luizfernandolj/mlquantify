from sklearn.linear_model import LogisticRegression
from mlquantify.mixture import DyS
from mlquantify.meta import QuaDapt
from sklearn.datasets import make_classification

def binary_dataset():
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    return X, y

X, y = binary_dataset()
learner = LogisticRegression()
# QuaDapt requires soft predictions
base_q = DyS(learner=learner) 
meta_q = QuaDapt(quantifier=base_q)
meta_q.fit(X, y)
preds = meta_q.predict(X)