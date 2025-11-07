from mlquantify.base import BaseQuantifier
import numpy as np

class MyQuantifier(BaseQuantifier):
    def __init__(self, param1=42, param2='default'):
        self.param1 = param1
        self.param2 = param2
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self
    
    def predict(self, X):
        _, counts = np.unique(self.classes_, return_counts=True)
        prevalence = counts / counts.sum()
        return prevalence


quantifier = MyQuantifier(param1=10, param2='custom')
print(quantifier.get_params())

X = np.random.rand(100, 10)
y = np.random.randint(0, 2, size=100)

print(quantifier.fit(X, y).predict(X))