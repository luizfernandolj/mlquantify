
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

@pytest.fixture(scope="session")
def binary_dataset():
    X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
    return X, y

@pytest.fixture(scope="session")
def multiclass_dataset():
    X, y = make_classification(n_samples=500, n_features=10, n_classes=3, n_clusters_per_class=1, random_state=42)
    return X, y

@pytest.fixture(params=["numpy", "pandas", "list"])
def binary_dataset_formats(request, binary_dataset):
    X, y = binary_dataset
    if request.param == "numpy":
        return X, y
    elif request.param == "pandas":
        return pd.DataFrame(X), pd.Series(y)
    elif request.param == "list":
        return X.tolist(), y.tolist()

@pytest.fixture(params=["numpy", "pandas", "list"])
def multiclass_dataset_formats(request, multiclass_dataset):
    X, y = multiclass_dataset
    if request.param == "numpy":
        return X, y
    elif request.param == "pandas":
        return pd.DataFrame(X), pd.Series(y)
    elif request.param == "list":
        return X.tolist(), y.tolist()
