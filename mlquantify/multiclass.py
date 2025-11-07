import numpy as np
from abc import abstractmethod
from mlquantify.base import BaseQuantifier
from mlquantify.utils._decorators import _fit_context
from mlquantify.base import BaseQuantifier


def define_binary(cls):
    original_fit = cls.fit
    original_predict = cls.predict

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y, *args, **kwargs):
        self.quantifiers_ = {}
        if self.strategy == 'ovr':
            self.quantifiers_ = _fit_ovr(self, original_fit, X, y, *args, **kwargs)
        elif self.strategy == 'ovo':
            self.quantifiers_ = _fit_ovo(self, original_fit, X, y, *args, **kwargs)
        else:
            return original_fit(self, X, y, *args, **kwargs)
        
    def predict(self, X, *args, **kwargs):
        if self.strategy == 'ovr':
            preds = _predict_ovr(self, original_predict, X, *args, **kwargs)
            return preds
        elif self.strategy == 'ovo':
            preds = _predict_ovo(self, original_predict, X, *args, **kwargs)
            return preds
        else:
            return original_predict(self, X, *args, **kwargs)

    cls.fit = fit
    cls.predict = predict
    return cls


@define_binary
class BinaryQuantifier(BaseQuantifier):
    def __init__(self, strategy='ovr'):
        self.strategy = strategy
        self.a = "a"
        
    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.target_input_tags.multi_class = False
        return tags

    def fit(self, X, y):
        # Fit padrão caso a estratégia não seja ovn ou ovo
        print("Fit padrão do quantificador")
        return self

    
    def predict(self, X):
        
        print("Predição usando BinaryQuantifier")
        print(self.a)
        return np.zeros(X.shape[0])  # Retorna zeros como placeholder
    


def _fit_ovr(quantifier, method, X, y):
        print("Fit One-vs-Rest")
        print(method)
        # implementar lógica ovR aqui
        return method(quantifier, X, y)

def _fit_ovo(quantifier, method, X, y):
    print("Fit One-vs-One")
    # implementar lógica ovo aqui
    return method(quantifier, X, y)
    
    
def _predict_ovr(quantifier, method, X):
    print("Predição One-vs-Rest")
    # implementar lógica ovR aqui
    return method(quantifier, X)

def _predict_ovo(quantifier, method, X):
    print("Predição One-vs-One")
    # implementar lógica ovo aqui
    return method(quantifier, X)
