
import pytest
import numpy as np

from mlquantify._config import config_context
from mlquantify.counting import ACC
from mlquantify.tree import (
    QuantificationTree,
    QuantificationForest,
    QuantificationTreeClassifier,
)


def _assert_valid_prevalence(prevalence, n_classes):
    prevalence = np.asarray(prevalence, dtype=float)
    assert prevalence.shape == (n_classes,)
    assert np.all(prevalence >= 0)
    assert np.all(prevalence <= 1)
    assert prevalence.sum() == pytest.approx(1.0)


def _tree_depth(node):
    if node.is_leaf():
        return 0
    return 1 + max(_tree_depth(node.left), _tree_depth(node.right))


# ---------------------------------------------------------------- classifier

@pytest.mark.parametrize("criterion", ["eb", "cqb"])
def test_classifier_fit_predict(binary_dataset, criterion):
    X, y = binary_dataset
    clf = QuantificationTreeClassifier(criterion=criterion, random_state=0).fit(X, y)
    preds = clf.predict(X)
    assert preds.shape == (len(y),)
    assert set(preds) <= set(clf.classes_)


@pytest.mark.parametrize("criterion", ["eb", "cqb"])
def test_classifier_predict_proba(multiclass_dataset, criterion):
    X, y = multiclass_dataset
    clf = QuantificationTreeClassifier(criterion=criterion, random_state=0).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_classifier_max_depth(binary_dataset):
    X, y = binary_dataset
    clf = QuantificationTreeClassifier(max_depth=1).fit(X, y)
    assert _tree_depth(clf.tree_) <= 1


def test_classifier_deterministic_with_seed(binary_dataset):
    X, y = binary_dataset
    p1 = QuantificationTreeClassifier(max_features="log2", random_state=7).fit(X, y).predict(X)
    p2 = QuantificationTreeClassifier(max_features="log2", random_state=7).fit(X, y).predict(X)
    np.testing.assert_array_equal(p1, p2)


def test_classifier_balances_errors(binary_dataset):
    # The split criterion targets |FP - FN| = 0 per class, so the CC estimate
    # on the training set should match the true training prevalence closely.
    X, y = binary_dataset
    clf = QuantificationTreeClassifier(criterion="eb").fit(X, y)
    preds = clf.predict(X)
    prev_true = np.array([(y == c).mean() for c in clf.classes_])
    prev_cc = np.array([(preds == c).mean() for c in clf.classes_])
    np.testing.assert_allclose(prev_cc, prev_true, atol=0.02)


def test_classifier_invalid_criterion(binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError):
        QuantificationTreeClassifier(criterion="gini").fit(X, y)


# ---------------------------------------------------------------- quantifiers

@pytest.mark.parametrize("quantifier_class", [QuantificationTree, QuantificationForest])
def test_quantifier_fit_predict_binary(quantifier_class, binary_dataset):
    X, y = binary_dataset
    q = quantifier_class(random_state=42)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=2)


@pytest.mark.parametrize("quantifier_class", [QuantificationTree, QuantificationForest])
def test_quantifier_fit_predict_multiclass(quantifier_class, multiclass_dataset):
    X, y = multiclass_dataset
    q = quantifier_class(random_state=42)
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=3)


def test_forest_subsample_and_parallel(binary_dataset):
    X, y = binary_dataset
    q = QuantificationForest(n_estimators=10, sample_fraction=0.7, n_jobs=2,
                             random_state=0)
    q.fit(X, y)
    assert len(q.estimator_) == 10
    assert q.tpr_.shape == (10, 2)
    assert q.fpr_.shape == (10, 2)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=2)


def test_forest_unadjusted(binary_dataset):
    X, y = binary_dataset
    q = QuantificationForest(n_estimators=5, adjusted=False, random_state=0)
    q.fit(X, y)
    # no adjustment: rates are the identity (tpr=1, fpr=0)
    np.testing.assert_array_equal(q.tpr_, 1.0)
    np.testing.assert_array_equal(q.fpr_, 0.0)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=2)


def test_forest_aggregate(binary_dataset):
    q = QuantificationForest()
    predictions = np.random.RandomState(0).randint(0, 2, size=(10, 200))
    with config_context(prevalence_return_type="array"):
        prevalence = q.aggregate(predictions)
    _assert_valid_prevalence(prevalence, n_classes=2)


def test_quantifier_set_params_propagates(binary_dataset):
    X, y = binary_dataset
    q = QuantificationTree().set_params(criterion="eb", max_depth=2).fit(X, y)
    assert q.estimator_.criterion == "eb"
    assert q.estimator_.max_depth == 2


def test_quantifier_get_params_roundtrip():
    q = QuantificationTree(criterion="eb", max_depth=3, random_state=1)
    params = q.get_params()
    q2 = QuantificationTree().set_params(**params)
    assert q2.get_params() == params


@pytest.mark.parametrize("quantifier_class", [QuantificationTree, QuantificationForest])
def test_quantifier_invalid_criterion(quantifier_class, binary_dataset):
    X, y = binary_dataset
    with pytest.raises(ValueError):
        quantifier_class(criterion="gini").fit(X, y)


def test_acc_composition(binary_dataset):
    # The paper's AC(Q) variant: adjusted count over a quantification tree.
    X, y = binary_dataset
    q = ACC(estimator=QuantificationTreeClassifier(random_state=0))
    q.fit(X, y)
    with config_context(prevalence_return_type="array"):
        prevalence = q.predict(X)
    _assert_valid_prevalence(prevalence, n_classes=2)
