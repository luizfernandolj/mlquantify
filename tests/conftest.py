import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture(scope="session")
def binary_dataset():
    """Generates a binary classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=2,
        weights=[0.6, 0.4],
        random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)

@pytest.fixture(scope="session")
def multiclass_dataset():
    """Generates a multiclass classification dataset."""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=10,
        weights=[0.3, 0.4, 0.3],
        random_state=42
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)

@pytest.fixture(scope="session")
def binary_classifier(binary_dataset):
    """Returns a trained binary classifier (LogisticRegression)."""
    X_train, X_test, y_train, y_test = binary_dataset
    clf = LogisticRegression(random_state=42, solver='liblinear')
    clf.fit(X_train, y_train)
    return clf

@pytest.fixture(scope="session")
def multiclass_classifier(multiclass_dataset):
    """Returns a trained multiclass classifier (RandomForestClassifier)."""
    X_train, X_test, y_train, y_test = multiclass_dataset
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)
    return clf

@pytest.fixture
def prob_predictions_binary(binary_classifier, binary_dataset):
    """Returns probabilistic predictions for the binary test set."""
    X_train, X_test, y_train, y_test = binary_dataset
    return binary_classifier.predict_proba(X_test)

@pytest.fixture
def predictions_binary(binary_classifier, binary_dataset):
    """Returns crisp predictions for the binary test set."""
    X_train, X_test, y_train, y_test = binary_dataset
    return binary_classifier.predict(X_test)
