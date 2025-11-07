from copy import deepcopy
import numpy as np
from abc import abstractmethod
from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import get_aggregation_requirements
from mlquantify.utils._decorators import _fit_context
from mlquantify.base import BaseQuantifier, MetaquantifierMixin
from mlquantify.utils._validation import validate_prevalences, check_has_method


def define_binary(cls):

    if check_has_method(cls, 'fit'):
        cls._original_fit = cls.fit
    if check_has_method(cls, 'predict'):
        cls._original_predict = cls.predict

    if check_has_method(cls, 'aggregate'):
        cls._original_aggregate = cls.aggregate
       
    cls.fit = BinaryQuantifier.fit 
    cls.predict = BinaryQuantifier.predict 
    cls.aggregate = BinaryQuantifier.aggregate
    return cls


def _fit_ovr(quantifier, X, y):
    # Implementação do ajuste One-vs-Rest
    classes = np.unique(y)
    quantifiers= {}
    for cls in classes:
        q = deepcopy(quantifier)
        y_binary = (y == cls).astype(int)
        q._original_fit(X, y_binary)
        quantifiers[cls] = q
    return quantifiers

def _fit_ovo(quantifier, X, y):
    # Implementação do ajuste One-vs-One
    from itertools import combinations
    classes = np.unique(y)
    quantifiers = {}
    for cls1, cls2 in combinations(classes, 2):
        q = deepcopy(quantifier)
        mask = (y == cls1) | (y == cls2)
        y_binary = y[mask]
        y_binary = (y_binary == cls1).astype(int)
        quantifier._original_fit(X, y_binary)
        quantifiers[(cls1, cls2)] = q
    return quantifiers

def _predict_ovr(quantifier, X):
    # Implementação da predição One-vs-Rest
    classes = quantifier.qtfs_.keys()
    preds = np.zeros(len(classes))
    for i, q in quantifier.qtfs_.items():
        preds[i] = q._original_predict(X)[1]
    return preds

def _predict_ovo(quantifier, X):
    # Implementação da predição One-vs-One
    from itertools import combinations
    classes = list(quantifier.qtfs_.keys())
    preds = np.zeros(len(classes))
    for i, (cls1, cls2) in enumerate(combinations(classes, 2)):
        q = quantifier.qtfs_[(cls1, cls2)]
        preds[i] = q._original_predict(X)[1]
    return preds

def _aggregate_ovr(quantifier, preds, y_train, train_preds=None):
    classes = np.unique(y_train)
    prevalences = {}
    
    for cls in classes:
        binary_preds = (preds[:, int(cls)])
        binary_preds = np.column_stack([1 - binary_preds, binary_preds])
        
        arguments = (binary_preds,)
        
        y_binary = (y_train == cls).astype(int)
        
        if train_preds is not None:
            binary_train_preds = (train_preds[:, int(cls)])
            binary_train_preds = np.column_stack([1 - binary_train_preds, binary_train_preds])
            arguments += (binary_train_preds,)
        arguments += (y_binary,)

        prevalences[int(cls)] = quantifier._original_aggregate(*arguments)[1]
    return prevalences

def _aggregate_ovo(quantifier, preds, y_train, train_preds=None):
    from itertools import combinations
    classes = np.unique(y_train)
    prevalences = {}
    
    for cls1, cls2 in combinations(classes, 2):
        binary_preds = preds[:, (cls1, cls2)]
        binary_preds = np.column_stack([1 - binary_preds, binary_preds])
        arguments = (binary_preds,)

        mask = (y_train == cls1) | (y_train == cls2)
        y_binary = y_train[mask]
        y_binary = (y_binary == cls1).astype(int)
        
        if train_preds is not None:
            binary_train_preds = train_preds[:, (cls1, cls2)]
            binary_train_preds = np.column_stack([1 - binary_train_preds, binary_train_preds])
            arguments += (binary_train_preds,)

        arguments += (y_binary,)

        prevalences[(cls1, cls2)] = quantifier._original_aggregate(*arguments)[1]
    return prevalences


class BinaryQuantifier(MetaquantifierMixin, BaseQuantifier):
    
    @_fit_context(prefer_skip_nested_validation=False)
    def fit(qtf, X, y):
        if len(np.unique(y)) <= 2:
            return qtf.fit(X, y)
        
        if not hasattr(qtf, 'strategy'):
            qtf.strategy = 'ovr'
        
        if qtf.strategy == 'ovr':
            qtf.qtfs_ = _fit_ovr(qtf, X, y)
        elif qtf.strategy == 'ovo':
            qtf.qtfs_ = _fit_ovo(qtf, X, y)
        else:
            raise ValueError("Strategy must be 'ovr' or 'ovo'")
        return qtf

    def predict(qtf, X):
        if hasattr(qtf, 'qtfs_'):
            if len(qtf.qtfs_) <= 2:
                preds = qtf._original_predict(X)
        
        if qtf.strategy == 'ovr':
            preds = _predict_ovr(qtf, X)
        elif qtf.strategy == 'ovo':
            preds = _predict_ovo(qtf, X)
        else:
            raise ValueError("Strategy must be 'ovr' or 'ovo'")

        preds = validate_prevalences(qtf, preds, qtf.qtfs_.keys())
        return preds

    def aggregate(qtf, *arguments):
        
        requirements = get_aggregation_requirements(qtf)
        if requirements.requires_train_proba and requirements.requires_train_labels:
            preds, train_preds, y_train = arguments
            arguments = {"preds": preds, "train_preds": train_preds, "y_train": y_train}
        elif requirements.requires_train_labels:
            preds, y_train = arguments
            arguments = {"preds": preds, "y_train": y_train}
        else:
            raise ValueError("Binary aggregation requires at least train labels")
        
        classes = np.unique(y_train)
        
        if not hasattr(qtf, 'strategy'):
            qtf.strategy = 'ovr'

        if len(classes) <= 2:
            return qtf._original_aggregate(*list(arguments.values()))

        
        if qtf.strategy == 'ovr':
            prevalences = _aggregate_ovr(qtf, **arguments)
        elif qtf.strategy == 'ovo':
            prevalences = _aggregate_ovo(qtf, **arguments)
        else:
            raise ValueError("Strategy must be 'ovr' or 'ovo'")

        prevalences = validate_prevalences(qtf, prevalences, classes)
        return prevalences