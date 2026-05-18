import numpy as np

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import (
    AggregationMixin,
    SoftLearnerQMixin,
)
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_data, validate_prevalences


class BaseComposeQuantifier(
    SoftLearnerQMixin,
    AggregationMixin,
    BaseQuantifier,
):
    r"""Base class for compose-based quantifiers.
 
    A compose quantifier combines:

    - a representation;
    - an objective;
    - a prevalence solver.
    """

    def __init__(
        self,
        representation,
        learner=None,
        solver="slsqp",
        aggregative=True,
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        self.representation = representation
        self.learner = learner
        self.solver = solver
        self.aggregative = aggregative
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

    def _validate_params(self):
        if self.aggregative:
            return super()._validate_params()

        return BaseQuantifier._validate_params(self)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False, sample_weight=None):
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        if self.aggregative:
            X_rep, y_rep = self._fit_learner_predictions(
                X,
                y,
                learner_fitted=learner_fitted,
            )
        else:
            X_rep, y_rep = X, y

        self.representation.fit(
            X_rep,
            y_rep,
            sample_weight=sample_weight,
        )
        
        self.train_priors_ = np.asarray([
            np.mean(y_rep == cls)
            for cls in self.classes_
        ])

        self.classes_ = self.representation.classes_
        self.train_representation_ = self.representation.class_representations_

        self.best_distance_ = None
        self.distances_ = None

        return self

    def predict(self, X):
        X = validate_data(self, X)

        if self.aggregative:
            X_rep = self._predict_learner(X)
        else:
            X_rep = X

        test_representation = self.representation.transform(X_rep)

        prevalences, distance = self._solve_prevalence(
            test_representation=test_representation,
            train_representation=self.train_representation_,
        )

        self.best_distance_ = distance

        return validate_prevalences(self, prevalences, self.classes_)

    def aggregate(
        self,
        test_representation,
        train_representation=None,
        train_labels=None,
        classes=None,
    ):
        if train_representation is not None and train_labels is not None:
            self.representation.fit(train_representation, train_labels)

            self.classes_ = (
                np.asarray(classes)
                if classes is not None
                else self.representation.classes_
            )

            self.train_representation_ = self.representation.class_representations_

        test_representation = self.representation.transform(test_representation)

        prevalences, distance = self._solve_prevalence(
            test_representation=test_representation,
            train_representation=self.train_representation_,
        )

        self.best_distance_ = distance

        return validate_prevalences(self, prevalences, self.classes_)

    def _solve_prevalence(self, test_representation, train_representation):
        raise NotImplementedError
