import numpy as np
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold


class _CVEstimator:
    """Internal cross-validated estimator helper.

    This class intentionally stays private: public quantifiers expose simple
    sklearn-like estimators while this object centralizes OOF predictions,
    cloning, refit and fold-ensemble behavior.
    """

    def __init__(
        self,
        estimator,
        cv=5,
        function="predict_proba",
        stratified=True,
        random_state=None,
        shuffle=True,
        cv_prediction="refit",
    ):
        self.estimator = estimator
        self.cv = cv
        self.function = function
        self.stratified = stratified
        self.random_state = random_state
        self.shuffle = shuffle
        self.cv_prediction = cv_prediction

    def fit_predict(self, X, y):
        if self.cv_prediction not in {"refit", "ensemble"}:
            raise ValueError("cv_prediction must be either 'refit' or 'ensemble'.")

        random_state = None if not self.shuffle else self.random_state
        splitter = self._make_splitter(random_state=random_state)

        predictions = []
        labels = []
        estimators = []

        for train_idx, test_idx in splitter.split(X, y):
            estimator = clone(self.estimator)
            estimator.fit(X[train_idx], y[train_idx])

            predictions.append(self._predict(estimator, X[test_idx]))
            labels.append(y[test_idx])
            estimators.append(estimator)

        self.estimators_ = estimators
        self.predictions_ = self._stack_predictions(predictions)
        self.labels_ = np.concatenate(labels)

        if self.cv_prediction == "ensemble":
            self.estimator_ = estimators
        else:
            self.estimator_ = self.estimator.fit(X, y)

        return self.predictions_, self.labels_, self.estimator_

    def _make_splitter(self, random_state):
        splitter_class = StratifiedKFold if self.stratified else KFold
        return splitter_class(
            n_splits=self.cv,
            shuffle=self.shuffle,
            random_state=random_state,
        )

    def _predict(self, estimator, X):
        if isinstance(self.function, str):
            if not hasattr(estimator, self.function):
                raise AttributeError(
                    f"The estimator does not have the method '{self.function}'."
                )
            return getattr(estimator, self.function)(X)

        if callable(self.function):
            return self.function(X)

        raise ValueError("The 'function' parameter must be a string or a callable.")

    def _stack_predictions(self, predictions):
        if self.function == "predict_proba":
            return np.vstack(predictions)

        return np.concatenate(predictions)
