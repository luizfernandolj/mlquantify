import numpy as np
from sklearn.neighbors import KernelDensity
from mlquantify.losses import MixtureNegativeLogLikelihoodLoss
from mlquantify.representations import KDERepresentation
from mlquantify.matching._base import BaseMatchingQuantifier
from mlquantify.matching._utils import gaussian_kernel
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._validation import validate_data, validate_prevalences, resolve_aggregate_classes
from mlquantify.base_aggregative import (
    AggregativeMixin,
    SoftPredictionMixin,
)

EPS = 1e-12

class KDEyQuantifier(
    SoftPredictionMixin,
    AggregativeMixin,
    BaseMatchingQuantifier,
):
    r"""Abstract base class for KDE-based density matching quantifiers.

    Fits kernel density estimates (KDEs) over classifier posterior probabilities
    for each class and estimates test prevalence by finding the mixture of class
    densities that best matches the test density.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bandwidth : float, default=0.1
        Bandwidth of the kernel density estimator.
    kernel : str, default='gaussian'
        Kernel type for the KDE. See ``sklearn.neighbors.KernelDensity``.
    solver : str, default='slsqp'
        Optimization solver.
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.matching._density import KDEyQuantifier
    >>> from mlquantify.base_aggregative import SoftPredictionMixin, AggregativeMixin
    >>> from mlquantify.matching._base import BaseMatchingQuantifier
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np
    >>> class MyKDEy(KDEyQuantifier):
    ...     def _precompute_density_terms(self, train_scores=None,
    ...                                   train_labels=None, train_representation=None):
    ...         pass  # no precomputation needed
    ...     def _solve_prevalence(self, test_representation, train_representations):
    ...         alpha = np.mean(test_representation)
    ...         return np.array([1 - alpha, alpha]), None
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> MyKDEy(estimator=LogisticRegression()).fit(X, y).predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Moreo, A., González, P., & del Coz, J. J. (2024).
               Kernel Density Estimation for Multiclass Quantification.
               *Machine Learning*, 113, 3075–3107.
    """

    _parameter_constraints = {
        "bandwidth": [Interval(0, None, inclusive_left=False)],
        "kernel": [Options(["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])],
        "solver": [Options(["auto", "slsqp"])],
        "cv": [Interval(2, None), None],
        "stratified": [bool],
        "shuffle": [bool],
        "random_state": ["random_state", None],
    }

    def __init__(
        self,
        estimator=None,
        bandwidth=0.1,
        kernel="gaussian",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        self.estimator = estimator
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.solver = solver
        self.cv = cv
        self.stratified = stratified
        self.shuffle = shuffle
        self.random_state = random_state

        super().__init__(
            representation=KDERepresentation(
                bandwidth=bandwidth,
                kernel=kernel,
            ),
            normalize=False,
        )

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = True
        tags.prediction_requirements.requires_train_labels = True
        return tags

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X,
        y,
        estimator_fitted=False,
        sample_weight=None,
        cv_prediction="refit",
    ):
        X, y = validate_data(self, X, y)

        train_scores, y_train = self._fit_estimator_predictions(
            X,
            y,
            estimator_fitted=estimator_fitted,
            cv_prediction=cv_prediction,
        )

        self._fit(
            train_scores,
            y_train,
            sample_weight=sample_weight,
        )

        self._precompute_density_terms(
            train_scores=train_scores,
            train_labels=y_train,
            train_representation=self.tr_representation_,
        )

        return self

    def predict(self, X):
        X = validate_data(self, X, ensure_2d=True)
        test_scores = self._predict_estimator(X)
        return self._predict(test_scores)

    def aggregate(self, test_scores, train_scores=None, y_train=None, classes=None):
        if train_scores is not None and y_train is not None:
            self._fit(train_scores, y_train)

            self._precompute_density_terms(
                train_scores=train_scores,
                train_labels=y_train,
                train_representation=self.tr_representation_,
            )

        if classes is not None:
            self.classes_ = resolve_aggregate_classes(self, classes, self.classes_)

        return self._predict(test_scores)

    def _precompute_density_terms(
        self,
        train_representation,
        train_labels,
        train_representations,
    ):
        raise NotImplementedError("Subclasses must implement _precompute_density_terms().")
    
    
    
    
    
    
class KDEyML(KDEyQuantifier):
    r"""KDEy Maximum Likelihood (KDEy-ML) quantifier.

    Targets prior probability shift. A multiclass distribution-matching method
    that replaces the per-class histograms of :class:`HDy` with multivariate
    kernel density estimates over the posterior vectors, then chooses the
    prevalence that maximises the mixture log-likelihood of the test
    posteriors. Native multiclass; scales where histogram matching fragments.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bandwidth : float, default=0.1
        Bandwidth of the kernel density estimator and the key parameter. Too
        small spikes each class density on its training points; too large
        blurs the class densities together.
    kernel : str, default='gaussian'
        Kernel shape placed on each training point (``'gaussian'``,
        ``'tophat'``, ``'epanechnikov'``, ``'exponential'``, ``'linear'``,
        ``'cosine'``).
    solver : str, default='slsqp'
        Constrained optimizer over the simplex (see :mod:`mlquantify.solvers`).
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    KDEy-ML is a maximum-likelihood method (compare :class:`~mlquantify.likelihood.EMQ`)
    and often outperforms EM when posteriors are recalibrated.

    See Also
    --------
    KDEyHD : Hellinger-distance KDE variant.
    KDEyCS : Closed-form Cauchy-Schwarz KDE variant.
    HDy : Histogram mixture model this generalises.

    Examples
    --------
    >>> from mlquantify.matching import KDEyML
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = KDEyML(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}
    >>> # call aggregate with pre-computed posterior scores
    >>> scores = q.estimator_.predict_proba(X)
    >>> q.aggregate(scores, scores, y)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Moreo, A., González, P., & del Coz, J. J. (2024).
               Kernel Density Estimation for Multiclass Quantification.
               *Machine Learning*, 113, 3075–3107.
    """


    def _precompute_density_terms(
        self,
        train_scores=None,
        train_labels=None,
        train_representation=None,
    ):
        pass

    def _solve_prevalence(self, test_representation, train_representations):
        class_kdes = train_representations

        class_likelihoods = np.asarray([
            np.exp(kde.score_samples(test_representation)) + EPS
            for kde in class_kdes
        ])
        loss_function = MixtureNegativeLogLikelihoodLoss(reduction="sum")

        def objective(prevalences):
            return loss_function(prevalences, class_likelihoods)

        return minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=self.solver,
        )
    
    
class KDEyHD(KDEyQuantifier):
    r"""KDEy Hellinger Distance (KDEy-HD) quantifier.

    Targets prior probability shift. A multiclass distribution-matching method
    that models each class with a multivariate KDE over the posterior vectors
    and selects the prevalence minimising the Hellinger distance between the
    test density and the mixture density, estimated by Monte-Carlo sampling.
    The KDE analogue of :class:`HDy` lifted to the simplex.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bandwidth : float, default=0.1
        Bandwidth of the kernel density estimator; controls density smoothness.
    kernel : str, default='gaussian'
        Kernel shape placed on each training point (``'gaussian'``,
        ``'tophat'``, ``'epanechnikov'``, ``'exponential'``, ``'linear'``,
        ``'cosine'``).
    montecarlo_trials : int, default=10000
        Number of Monte Carlo samples used to approximate the Hellinger
        distance integral; more samples reduce approximation noise.
    solver : str, default='slsqp'
        Constrained optimizer over the simplex (see :mod:`mlquantify.solvers`).
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    See Also
    --------
    KDEyML : Maximum-likelihood KDE variant.
    KDEyCS : Closed-form Cauchy-Schwarz KDE variant.

    Examples
    --------
    >>> from mlquantify.matching import KDEyHD
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = KDEyHD(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Moreo, A., González, P., & del Coz, J. J. (2024).
               Kernel Density Estimation for Multiclass Quantification.
               *Machine Learning*, 113, 3075–3107.
    """

    _parameter_constraints = {
        "bandwidth": [Interval(0, None, inclusive_left=False)],
        "kernel": [Options(["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"])],
        "montecarlo_trials": [Interval(1, None)],
        "random_state": ["random_state", None],
    }

    def __init__(
        self,
        estimator=None,
        bandwidth=0.1,
        kernel="gaussian",
        montecarlo_trials=10000,
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        self.montecarlo_trials = montecarlo_trials

        super().__init__(
            estimator=estimator,
            bandwidth=bandwidth,
            kernel=kernel,
            solver=solver,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )

    def _precompute_density_terms(
        self,
        train_scores=None,
        train_labels=None,
        train_representation=None,
    ):
        n_classes = len(self.classes_)
        samples_per_class = max(
            1,
            int(self.montecarlo_trials // n_classes),
        )

        samples = []

        for kde in train_representation:
            samples.append(
                kde.sample(
                    samples_per_class,
                    random_state=self.random_state,
                )
            )

        self.reference_samples_ = np.vstack(samples)

        self.reference_classwise_density_ = np.asarray([
            np.exp(kde.score_samples(self.reference_samples_)) + EPS
            for kde in train_representation
        ])

        self.reference_density_ = (
            np.mean(self.reference_classwise_density_, axis=0) + EPS
        )

    def _solve_prevalence(self, test_representation, train_representations):
        test_kde = KernelDensity(
            bandwidth=self.bandwidth,
            kernel=self.kernel,
        ).fit(test_representation)

        test_density = (
            np.exp(test_kde.score_samples(self.reference_samples_)) + EPS
        )

        importance_weights = test_density / self.reference_density_
        class_ratios = self.reference_classwise_density_ / test_density

        def objective(prevalences):
            density_ratio = prevalences @ class_ratios
            values = ((np.sqrt(density_ratio) - 1.0) ** 2) * importance_weights
            return float(np.mean(values))

        return minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=self.solver,
        )
    
    
    
    
    
    
class KDEyCS(KDEyQuantifier):
    r"""KDEy Cauchy-Schwarz (KDEy-CS) quantifier.

    Targets prior probability shift. A multiclass distribution-matching method
    that models each class with a Gaussian KDE over the posterior vectors and
    selects the prevalence minimising the Cauchy-Schwarz divergence, which has
    a closed form. The most efficient KDEy variant, since the expensive
    train-train kernel terms are precomputed once at fit time.

    Parameters
    ----------
    estimator : estimator, optional
        A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
    bandwidth : float, default=0.1
        Bandwidth of the Gaussian kernel density estimator; controls smoothness.
    kernel : str, default='gaussian'
        Must be ``'gaussian'``; the closed-form divergence is Gaussian-specific.
    solver : str, default='slsqp'
        Constrained optimizer over the simplex (see :mod:`mlquantify.solvers`).
    cv : int or None, default=None
        Cross-validation folds for computing training scores.
    stratified : bool, default=True
        Whether to stratify CV splits.
    shuffle : bool, default=False
        Whether to shuffle data before splitting.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        The fitted underlying classifier.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    The closed form makes KDEy-CS the fastest KDEy variant: the train-train
    kernel matrices are computed once at training time and reused for any test
    bag.

    See Also
    --------
    KDEyML : Maximum-likelihood KDE variant.
    KDEyHD : Hellinger-distance KDE variant.

    Examples
    --------
    >>> from mlquantify.matching import KDEyCS
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = KDEyCS(estimator=LogisticRegression()).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Moreo, A., González, P., & del Coz, J. J. (2024).
               Kernel Density Estimation for Multiclass Quantification.
               *Machine Learning*, 113, 3075–3107.
    """

    _parameter_constraints = {
        "bandwidth": [Interval(0, None, inclusive_left=False)],
        "kernel": [Options(["gaussian"])],
    }

    def __init__(
        self,
        estimator=None,
        bandwidth=0.1,
        kernel="gaussian",
        solver="slsqp",
        cv=None,
        stratified=True,
        shuffle=False,
        random_state=None,
    ):
        if kernel != "gaussian":
            raise ValueError("KDEyCS supports only gaussian KDE.")

        super().__init__(
            estimator=estimator,
            bandwidth=bandwidth,
            kernel=kernel,
            solver=solver,
            cv=cv,
            stratified=stratified,
            shuffle=shuffle,
            random_state=random_state,
        )

    def _precompute_density_terms(
        self,
        train_scores=None,
        train_labels=None,
        train_representation=None,
    ):
        X = self.representation.transform(train_scores)
        y = np.asarray(train_labels)

        self.centers_ = [X[y == cls] for cls in self.classes_]

        self.class_counts_ = np.asarray(
            [max(1, len(center)) for center in self.centers_],
            dtype=float,
        )

        self.effective_bandwidth_ = np.sqrt(2.0) * self.bandwidth

        gram = np.zeros((len(self.classes_), len(self.classes_)))

        for i, left in enumerate(self.centers_):
            for j, right in enumerate(self.centers_[i:], start=i):
                value = np.sum(
                    gaussian_kernel(
                        left,
                        right,
                        bandwidth=self.effective_bandwidth_,
                    )
                )
                gram[i, j] = gram[j, i] = value

        self.train_gram_ = gram

    def _solve_prevalence(self, test_representation, train_representations):
        test_representation = self.representation.transform(test_representation)

        train_test_kernel_sums = np.asarray([
            np.sum(
                gaussian_kernel(
                    centers,
                    test_representation,
                    bandwidth=self.effective_bandwidth_,
                )
            )
            for centers in self.centers_
        ])

        test_weight = 1.0 / max(1, test_representation.shape[0])

        def objective(prevalences):
            prevalences = np.clip(prevalences, EPS, None)
            prevalences = prevalences / prevalences.sum()

            weighted_prevalences = prevalences / (self.class_counts_ + EPS)

            train_test_term = (
                np.dot(weighted_prevalences, train_test_kernel_sums)
                * test_weight
            )

            train_train_term = (
                weighted_prevalences
                @ ((self.train_gram_ + EPS) @ weighted_prevalences)
            )

            return float(
                -np.log(train_test_term + EPS)
                + 0.5 * np.log(train_train_term + EPS)
            )

        return minimize_prevalence(
            objective=objective,
            n_classes=len(self.classes_),
            solver=self.solver,
        )
