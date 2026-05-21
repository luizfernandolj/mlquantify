import numpy as np

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import (
    is_aggregative_quantifier,
    uses_crisp_predictions,
    uses_soft_predictions,
)
from mlquantify.representations import PredictionRepresentation
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_data, validate_prevalences


class BaseComposeQuantifier(
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
    ):
        self.representation = representation
        self.learner = learner
        self.solver = solver
        self.aggregative = aggregative

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()

        if not getattr(self, "aggregative", True):
            tags.has_estimator = False
            tags.estimator_function = None
            tags.estimator_type = None

        return tags

    def _is_configured_aggregative(self):
        return bool(getattr(self, "aggregative", True))

    def _check_aggregation_configuration(self):
        if not self._is_configured_aggregative():
            return

        if not is_aggregative_quantifier(self):
            raise ValueError(
                f"{self.__class__.__name__} was configured with aggregative=True, "
                "so the concrete method must inherit AggregationMixin."
            )

        if not (uses_soft_predictions(self) or uses_crisp_predictions(self)):
            raise ValueError(
                f"{self.__class__.__name__} was configured with aggregative=True, "
                "so the concrete method must inherit either SoftLearnerQMixin "
                "or CrispLearnerQMixin."
            )

        if not isinstance(self.representation, PredictionRepresentation):
            raise ValueError(
                f"{self.__class__.__name__} was configured with aggregative=True, "
                "so its representation must be a PredictionRepresentation."
            )

    def _uses_learner_predictions(self):
        self._check_aggregation_configuration()
        return self._is_configured_aggregative()

    def _validate_params(self):
        self._check_aggregation_configuration()
        return super()._validate_params()

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        learner_fitted=False,
        sample_weight=None,
        cv_prediction="refit",
    ):
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        if self._uses_learner_predictions():
            X_rep, y_rep = self._fit_learner_predictions(
                X,
                y,
                learner_fitted=learner_fitted,
                cv_prediction=cv_prediction,
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

        if self._uses_learner_predictions():
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
