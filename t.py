import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from mlquantify.likelihood import EMQ
from mlquantify.utils import get_prev_from_labels
from quapy.method.aggregative import EMQ as EMQ_QP


# -------------------------------------------------
# Dataset
# -------------------------------------------------
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_classes=3,
    n_informative=5,
    random_state=20
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

print("True test prevalence:", get_prev_from_labels(y_test))


# -------------------------------------------------
# Train classifier
# -------------------------------------------------
clf = RandomForestClassifier(random_state=20)
clf.fit(X_train, y_train)

train_probs = clf.predict_proba(X_train)
probs = clf.predict_proba(X_test)


# -------------------------------------------------
# Artificial miscalibration
# -------------------------------------------------
def distort_probs(probs, alpha=3.0):
    """Make probabilities overconfident."""
    probs = np.power(probs, alpha)
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs

miscalibrated_probs = distort_probs(probs, alpha=6.0)


# -------------------------------------------------
# Pass scores directly to EMQ.aggregate
# -------------------------------------------------
for calib in ["ts", "bcts", "nbvs", "vs", None]:
    emq = EMQ(learner=clf, calib_function=calib)
    emq_qp = EMQ_QP(RandomForestClassifier(random_state=20), calib=calib)
    # NOTE: pass miscalibrated scores directly
    preds = emq.aggregate(miscalibrated_probs, train_probs, y_train)
    emq_qp.fit(X_train, y_train)
    preds_qp = emq_qp.predict(X_test)

    print(f"{calib} →", preds)
    print(f"{calib} →", preds_qp)