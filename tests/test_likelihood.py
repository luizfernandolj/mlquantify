import pytest
import numpy as np
from mlquantify.likelihood import EMQ

def test_EMQ_binary(binary_classifier, binary_dataset):
    X_train, X_test, y_train, y_test = binary_dataset
    q = EMQ(learner=binary_classifier)
    q.fit(X_train, y_train)
    prev = q.predict(X_test)
    assert isinstance(prev, dict)
    assert len(prev) == 2
    assert pytest.approx(sum(prev.values())) == 1.0

def test_EMQ_aggregate_static():
    # Test the static EM method directly
    posteriors = np.array([[0.8, 0.2], [0.6, 0.4], [0.1, 0.9]])
    priors = np.array([0.5, 0.5])
    qs, _ = EMQ.EM(posteriors, priors)
    assert len(qs) == 2
    assert pytest.approx(sum(qs)) == 1.0
