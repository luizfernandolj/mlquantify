import numpy as np
from abc import abstractmethod
from sklearn.neighbors import KernelDensity

from mlquantify.utils._decorators import _fit_context
from mlquantify.base import BaseQuantifier
from mlquantify.utils import validate_y, validate_predictions, validate_data
from mlquantify.base_aggregative import AggregationMixin, SoftLearnerQMixin, _get_learner_function
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._validation import validate_prevalences

EPS = 1e-12

class BaseKDE(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    """
    Classe base para métodos KDEy.
    Treina uma densidade por classe no espaço de probabilidades preditas.
    """

    _parameter_constraints = {
        "bandwidth": [Interval(0, None, inclusive_right=False)],
        "kernel": [Options(["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])],
    }

    def __init__(self, learner=None, bandwidth: float = 0.1, kernel: str = "gaussian"):
        self.learner = learner
        self.bandwidth = bandwidth
        self.kernel = kernel
        self._precomputed = False
        self.best_distance = None

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted=False):
        """Treina o quantificador com validação cruzada interna."""
        X, y = validate_data(self, X, y, ensure_2d=True, ensure_min_samples=2)
        validate_y(self, y)
        
        self.classes = np.unique(y)
        
        learner_function = _get_learner_function(self)

        if learner_fitted:
            train_predictions = getattr(self.learner, learner_function)(X)
            train_y_values = y
        else:
            train_predictions, train_y_values = apply_cross_validation(
                self.learner, X, y,
                function=learner_function, cv=5,
                stratified=True, shuffle=True
            )

        self.train_predictions = train_predictions
        self.train_y_values = train_y_values
        self._precompute_training(train_predictions, train_y_values)
        return self

    def _fit_kde_models(self, train_predictions, train_y_values):
        """Ajusta um KDE para cada classe."""
        P = np.atleast_2d(train_predictions)
        y = np.asarray(train_y_values)
        classes = np.unique(y)
        self._class_kdes = []

        for c in classes:
            Xi = P[y == c]
            if Xi.shape[0] == 0:
                Xi = np.ones((1, P.shape[1])) / P.shape[1]  # fallback
            kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
            kde.fit(Xi)
            self._class_kdes.append(kde)

        self._precomputed = True

    def predict(self, X):
        predictions = getattr(self.learner, _get_learner_function(self))(X)
        return self.aggregate(predictions, self.train_predictions, self.train_y_values)
    
    def aggregate(self, predictions, train_predictions, train_y_values):
        """Agrega as previsões em estimativas de prevalência."""
        predictions = validate_predictions(self, predictions)
        
        self.classes = np.unique(train_y_values) if not hasattr(self, 'classes') else self.classes
        
        if not self._precomputed:
            self._precompute_training(train_predictions, train_y_values)
            self._precomputed = True

        prevalence, _ = self._solve_prevalences(predictions)
        prevalence = np.clip(prevalence, EPS, None)
        prevalence = validate_prevalences(self, prevalence, self.classes)
        return prevalence

    def best_distance(self):
        """Retorna a melhor distância encontrada durante o ajuste."""
        if self.best_distance is not None:
            return self.best_distance
        _, best_distance = self._solve_prevalences(self.train_predictions)
        return best_distance

    @abstractmethod
    def _precompute_training(self, train_predictions, train_y_values):
        """Subclasses podem pré-computar estatísticas de treinamento aqui."""
        raise NotImplementedError

    @abstractmethod
    def _solve_prevalences(self, predictions, train_predictions, train_y_values):
        """Subclasses devem implementar o cálculo de α."""
        raise NotImplementedError