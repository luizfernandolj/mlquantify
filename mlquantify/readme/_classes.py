import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from mlquantify.base import BaseQuantifier
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._validation import validate_data, validate_prevalences


def _solve_subset(profiles_labeled, profiles_unlabeled, y_idx, n_classes, solver):
    """Estimate P(D) from one feature subset by constrained least squares.

    ``profiles_labeled`` / ``profiles_unlabeled`` are integer profile keys.
    Builds the conditional profile distribution matrix A (P(S|D), one column
    per class) and the unlabeled profile distribution b (P(S)), then solves
    ``min ||A pi - b||^2`` on the simplex.
    """
    all_profiles, labeled_codes = np.unique(
        np.concatenate([profiles_labeled, profiles_unlabeled]),
        return_inverse=True,
    )
    n_profiles = len(all_profiles)
    codes_labeled = labeled_codes[: len(profiles_labeled)]
    codes_unlabeled = labeled_codes[len(profiles_labeled):]

    A = np.zeros((n_profiles, n_classes))
    for j in range(n_classes):
        counts = np.bincount(codes_labeled[y_idx == j], minlength=n_profiles)
        total = counts.sum()
        if total == 0:
            return None
        A[:, j] = counts / total
    b = np.bincount(codes_unlabeled, minlength=n_profiles) / len(codes_unlabeled)

    def objective(prevalences):
        residual = A @ prevalences - b
        return float(residual @ residual)

    try:
        prevalences, _ = minimize_prevalence(objective, n_classes, solver=solver)
    except Exception:
        return None
    return prevalences


class ReadMe(BaseQuantifier):
    r"""ReadMe quantifier (Hopkins & King, 2010).

    Targets prior probability shift. A **non-aggregative** quantifier: it uses
    no classifier at all. It solves the accounting identity

    .. math::

        P(S) = P(S \mid D) \, P(D)

    directly, where :math:`S` is the joint distribution of binary feature
    profiles and :math:`D` the class: :math:`P(S)` is tabulated on the test
    (unlabeled) set, :math:`P(S \mid D)` on the training (labeled) set, and
    the class prevalences :math:`P(D)` are obtained by least squares
    constrained to the probability simplex. Because :math:`2^K` profiles are
    intractable for many features, the estimation is repeated on many small
    random feature subsets — a form of kernel smoothing (King & Lu, 2008) —
    and the estimates averaged.

    Parameters
    ----------
    n_subsets : int, default=50
        Number of random feature subsets to estimate on.
    subset_size : int, default=15
        Number of features per subset (the papers use 5–25). Capped at the
        number of available features and at 25 (profile keys are ``int64``).
    binarize : {'auto', True, False}, default='auto'
        The profile tabulation requires binary features. ``'auto'`` thresholds
        each feature at its labeled-set median unless the input is already
        0/1; ``True`` always thresholds; ``False`` requires binary input and
        raises otherwise.
    variance_weighting : bool, default=True
        Sample features into subsets with probability proportional to their
        labeled-set variance (more informative features are drawn more often),
        as in the reference implementation.
    solver : {'slsqp', 'grid', 'ternary', 'bounded', 'auto'}, default='slsqp'
        Simplex solver passed to :func:`~mlquantify.solvers.minimize_prevalence`.
    n_jobs : int or None, default=None
        Number of parallel jobs for the subset loop.
    random_state : int or None, default=None
        Seed for the feature subset draws.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    thresholds_ : ndarray of shape (n_features,) or None
        Per-feature binarization thresholds (labeled medians); ``None`` when
        the input is used as-is.
    feature_weights_ : ndarray of shape (n_features,)
        Subset sampling probabilities.

    Notes
    -----
    The only assumption connecting the two samples is that the
    class-conditional feature distribution is stable,
    :math:`P(S \mid D)_{\text{train}} = P(S \mid D)_{\text{test}}`. Neither
    the class distribution :math:`P(D)` nor the marginal feature distribution
    :math:`P(S)` needs to be the same in the two sets, which is what makes the
    method robust to prior shift without any classifier or correction step.
    ReadMe was designed for word-stem indicators; with continuous features the
    median binarization discards information — consider :class:`ReadMe2`.

    See Also
    --------
    ReadMe2 : Continuous-feature successor with learned projections.

    Examples
    --------
    >>> from mlquantify.readme import ReadMe
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=500, random_state=42)
    >>> q = ReadMe(n_subsets=20, subset_size=10, random_state=0).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Hopkins, D. J., & King, G. (2010). A Method of Automated
               Nonparametric Content Analysis for Social Science.
               *American Journal of Political Science*, 54(1), 229-247.
        .. [2] King, G., & Lu, Y. (2008). Verbal Autopsy Methods with
               Multiple Causes of Death. *Statistical Science*, 23(1), 78-91.
    """

    _parameter_constraints = {
        "n_subsets": [Interval(1, None, inclusive_right=False)],
        "subset_size": [Interval(1, 25)],
        "binarize": [Options(["auto", True, False])],
        "variance_weighting": [bool],
        "solver": [Options(["auto", "slsqp", "grid", "ternary", "bounded"])],
        "n_jobs": [Interval(-1, None, inclusive_right=False), type(None)],
        "random_state": [Interval(0, None, inclusive_right=False), type(None)],
    }

    def __init__(self,
                 n_subsets=50,
                 subset_size=15,
                 binarize="auto",
                 variance_weighting=True,
                 solver="slsqp",
                 n_jobs=None,
                 random_state=None):
        self.n_subsets = n_subsets
        self.subset_size = subset_size
        self.binarize = binarize
        self.variance_weighting = variance_weighting
        self.solver = solver
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _is_binary(self, X):
        return np.isin(X, (0, 1)).all()

    def fit(self, X, y):
        r"""Store the labeled profile source and binarization thresholds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Labeled features (binary indicators, or continuous when
            ``binarize`` is enabled).
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : ReadMe
            Fitted quantifier.
        """
        self._validate_params()
        X, y = validate_data(self, X, y)
        y = np.asarray(y)
        self.classes_, self._y_idx = np.unique(y, return_inverse=True)

        binarize = self.binarize
        if binarize == "auto":
            binarize = not self._is_binary(X)
        if binarize:
            self.thresholds_ = np.median(X, axis=0)
            X_bin = (X > self.thresholds_).astype(np.int8)
        else:
            if not self._is_binary(X):
                raise ValueError(
                    "ReadMe requires binary (0/1) features when binarize=False; "
                    "pass binarize='auto' or True to threshold at the labeled median."
                )
            self.thresholds_ = None
            X_bin = X.astype(np.int8)
        self._X_bin = X_bin

        variances = X_bin.var(axis=0)
        if self.variance_weighting and variances.sum() > 0:
            self.feature_weights_ = variances / variances.sum()
        else:
            self.feature_weights_ = np.full(X.shape[1], 1.0 / X.shape[1])
        return self

    def predict(self, X):
        r"""Estimate class prevalences on the given (unlabeled) data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        ndarray of shape (n_classes,)
            Class prevalence estimates.
        """
        X = validate_data(self, X)
        if self.thresholds_ is not None:
            X_bin = (X > self.thresholds_).astype(np.int8)
        else:
            X_bin = X.astype(np.int8)

        n_features = self._X_bin.shape[1]
        subset_size = min(self.subset_size, n_features)
        n_nonzero = np.count_nonzero(self.feature_weights_)
        subset_size = min(subset_size, n_nonzero) if n_nonzero else subset_size

        rng = check_random_state(self.random_state)
        powers = 1 << np.arange(subset_size, dtype=np.int64)
        subsets = [
            rng.choice(n_features, size=subset_size, replace=False,
                       p=self.feature_weights_)
            for _ in range(self.n_subsets)
        ]

        estimates = Parallel(n_jobs=self.n_jobs)(
            delayed(_solve_subset)(
                self._X_bin[:, subset].astype(np.int64) @ powers,
                X_bin[:, subset].astype(np.int64) @ powers,
                self._y_idx,
                len(self.classes_),
                self.solver,
            )
            for subset in subsets
        )
        estimates = [est for est in estimates if est is not None]
        if not estimates:
            raise RuntimeError(
                "ReadMe could not estimate prevalences on any feature subset."
            )

        prevalences = np.mean(estimates, axis=0)
        prevalences = prevalences / prevalences.sum()
        return validate_prevalences(self, prevalences, self.classes_)
