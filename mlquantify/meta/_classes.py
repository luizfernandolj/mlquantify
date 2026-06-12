from mlquantify.utils import check_classes_attribute
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split

from mlquantify.base import BaseQuantifier, MetaquantifierMixin
from mlquantify.metrics._slq import MSE
from mlquantify.matching import SORD, DyS
from mlquantify.matching._utils import get_histogram
from mlquantify.losses import get_loss
from mlquantify.metrics import hellinger
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils import Options, Interval
from mlquantify.utils import _fit_context
from mlquantify.confidence import (
    construct_confidence_region
)
from mlquantify.base_aggregative import (
    _get_estimator_function,
    is_aggregative_quantifier,
    get_aggregation_requirements,
    uses_soft_predictions
)
from mlquantify.utils._sampling import (
    bootstrap_sample_indices
)
from mlquantify.model_selection import APP, NPP, UPP
from mlquantify.utils._validation import validate_data, validate_prevalences
from mlquantify.utils.prevalence import get_prev_from_labels
from mlquantify.utils import check_random_state
from mlquantify._config import config_context
from mlquantify.multiclass import binary_quantifier


def getHist(values, n_bins):
    """Compute a concatenated histogram over all columns of ``values``.

    Parameters
    ----------
    values : array-like of shape (n_samples,) or (n_samples, n_features)
        Score matrix. 1-D inputs are reshaped to a single column.
    n_bins : int
        Number of equal-width histogram bins.

    Returns
    -------
    hist : ndarray
        Concatenated histogram across all feature columns, normalised to
        sum to 1 within each column.
    """
    values = np.asarray(values, dtype=float)

    if values.ndim == 1:
        values = values.reshape(-1, 1)

    return np.concatenate([
        get_histogram(values[:, feature_idx], n_bins)
        for feature_idx in range(values.shape[1])
    ])



def get_protocol_sampler(protocol_name, batch_size, n_prevalences, min_prev, max_prev, n_classes):
    r""" Returns a prevalence sampler function based on the specified protocol name.

    Parameters
    ----------
    protocol_name : str
        The name of the protocol ('app', 'npp', 'upp', 'upp-k').
    batch_size : int
        The size of each batch.
    n_prevalences : int
        The number of prevalences to sample.
    min_prev : float
        The minimum prevalence value.
    max_prev : float
        The maximum prevalence value.
    n_classes : int
        The number of classes.

    Returns
    -------
    callable
        A function that generates prevalence samples according to the specified protocol.
    """

    if protocol_name == 'artificial':
        protocol = APP(batch_size=batch_size,
                           n_prevalences=n_prevalences,
                           min_prev=min_prev,
                           max_prev=max_prev)

    elif protocol_name == 'natural':
        protocol = NPP(batch_size=batch_size,
                           n_samples=n_prevalences)

    elif protocol_name == 'uniform':
            protocol = UPP(batch_size=batch_size,
                           n_prevalences=n_prevalences,
                           algorithm='uniform',
                           min_prev=min_prev,
                           max_prev=max_prev)
    elif protocol_name == 'kraemer':
        protocol = UPP(batch_size=batch_size,
                           n_prevalences=n_prevalences,
                           algorithm='kraemer',
                           min_prev=min_prev,
                           max_prev=max_prev)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}")
    return protocol

class EnsembleQ(MetaquantifierMixin, BaseQuantifier):
    r"""Ensemble Quantifier with prevalence-controlled diversity.

    Targets prior probability shift, including shifts whose magnitude is
    unknown at training time. Trains many copies of a base quantifier on
    subsamples drawn at deliberately different class prevalences, then
    aggregates their estimates, optionally keeping only the members whose
    training distribution resembles the test sample (dynamic selection). The
    spread of training prevalences brackets the possible test distributions.

    Parameters
    ----------
    quantifier : BaseQuantifier
        The base quantifier replicated across ensemble members.
    size : int, default=50
        Number of ensemble members to train.
    min_prop : float, default=0.1
        Minimum class prevalence proportion for sampling batches.
    max_prop : float, default=1.0
        Maximum class prevalence proportion for sampling batches; together with
        ``min_prop`` it sets the diversity of training prevalences.
    selection_metric : {'all', 'ptr', 'ds'}, default='all'
        Which members vote at prediction time.

        - ``'all'`` : use every member (a plain bagged average).
        - ``'ptr'`` : keep members whose training prevalence is closest to an
          initial test estimate.
        - ``'ds'`` : keep members whose training score distribution is closest
          to the test distribution.
    p_metric : float, default=0.25
        Fraction of ensemble members to retain when a selection metric is used.
    protocol : {'artificial', 'natural', 'uniform', 'kraemer'}, default='uniform'
        Prevalence-sampling protocol for generating training batches.
    return_type : {'mean', 'median'}, default='mean'
        Aggregation function applied to the selected member estimates.
    max_sample_size : int or None, default=None
        Maximum samples per training batch; ``None`` uses the full dataset.
    max_trials : int, default=100
        Maximum sampling attempts per batch.
    n_jobs : int, default=1
        Number of parallel jobs for training.
    verbose : bool, default=False
        Print progress messages.

    Attributes
    ----------
    models : list
        Fitted ensemble member quantifiers.
    train_prevalences : list
        Training prevalences for each ensemble member.
    classes : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    Members are trained with sampling-with-replacement so that ``p(x|y)`` is
    preserved while only ``p(y)`` varies. Dynamic selection (``'ptr'``/``'ds'``)
    is what specialises the ensemble to each test bag; with ``'all'`` it reduces
    to a bagged average. Wraps any base quantifier.

    See Also
    --------
    AggregativeBootstrap : Resampling wrapper for confidence regions.
    QuaDapt : Drift-resilient adaptation wrapper.

    Examples
    --------
    >>> from mlquantify.meta import EnsembleQ
    >>> from mlquantify.matching import DyS
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=300, random_state=42)
    >>> q = EnsembleQ(DyS(estimator=LogisticRegression()), size=10).fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}

    References
    ----------
    .. dropdown:: References

        .. [1] Pérez-Gállego, P., Quevedo, J. R., & del Coz, J. J. (2017).
               Using Ensembles for Problems with Characterizable Changes in Data
               Distribution: A Case Study on Quantification.
               *Information Fusion*, 34, 87–100.
        .. [2] Pérez-Gállego, P., Castaño, A., Quevedo, J. R., & del Coz, J. J.
               (2019). Dynamic Ensemble Selection for Quantification Tasks.
               *Information Fusion*, 45, 1–15.
    """

    _parameter_constraints = {
        "quantifier": [BaseQuantifier],
        "size": [Interval(left=1, right=None, discrete=True)],
        "min_prop": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "max_prop": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "selection_metric": [Options(['all', 'ptr', 'ds'])],
        "p_metric": [Interval(left=0.0, right=1.0, inclusive_left=True, inclusive_right=True)],
        "protocol": [Options(['artificial', 'natural', 'uniform', 'kraemer'])],
        "return_type": [Options(['mean', 'median'])],
        "max_sample_size": [Interval(left=1, right=None, discrete=True), None],
        "max_trials": [Interval(left=1, right=None, discrete=True)],
        "n_jobs": [Interval(left=-1, right=None, discrete=True)],
        "verbose": [bool],
    }

    def __init__(self,
                 quantifier,
                 size=50,
                 min_prop=0.1,
                 max_prop=1,
                 selection_metric='all',
                 protocol="uniform",
                 p_metric=0.25,
                 return_type="mean",
                 max_sample_size=None,
                 max_trials=100,
                 n_jobs=1,
                 verbose=False):

        self.quantifier = quantifier
        self.size = size
        self.min_prop = min_prop
        self.max_prop = max_prop
        self.p_metric = p_metric
        self.protocol = protocol
        self.selection_metric = selection_metric
        self.return_type = return_type
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_sample_size = max_sample_size
        self.max_trials = max_trials

    def sout(self, msg):
        """Prints a message if verbose is True."""
        if self.verbose:
            print('[Ensemble]' + msg)

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit the ensemble by training one base quantifier per sampled batch.

        Batches are drawn from ``(X, y)`` according to the chosen ``protocol``
        so that each member is trained on a subset with a different class
        prevalence distribution, promoting diversity. When ``selection_metric``
        is ``'ds'``, posterior probabilities are precomputed for later use
        during prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : EnsembleQ
            The fitted ensemble quantifier.

        Raises
        ------
        ValueError
            If ``selection_metric='ds'`` is used on a multiclass dataset.

        Examples
        --------
        >>> from mlquantify.meta import EnsembleQ
        >>> from mlquantify.matching import DyS
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=300, random_state=42)
        >>> q = EnsembleQ(DyS(estimator=LogisticRegression()), size=10).fit(X, y)
        """
        self.sout('Fit')

        self.models = []
        self.train_prevalences = []
        self.train_distributions = []
        self.posteriors_generator = []

        self.classes = np.unique(y)
        X, y = validate_data(self, X, y)

        if self.selection_metric == 'ds' and not len(self.classes) == 2:
            raise ValueError(f'ds selection_metric is only defined for binary quantification, but this dataset is not binary')
        # randomly chooses the prevalences for each member of the ensemble (preventing classes with less than
        # min_pos positive examples)
        sample_size = len(y) if self.max_sample_size is None else min(self.max_sample_size, len(y))

        protocol = get_protocol_sampler(
            protocol_name=self.protocol,
            batch_size=sample_size,
            n_prevalences=self.size,
            min_prev=self.min_prop,
            max_prev=self.max_prop,
            n_classes=len(self.classes)
        )

        posteriors = None
        if self.selection_metric == 'ds':
            # precompute the training posterior probabilities
            posteriors, self.posteriors_generator = self.ds_get_posteriors(X, y)

        for idx in protocol.split(X, y):
            X_batch, y_batch = X[idx], y[idx]
            model = deepcopy(self.quantifier)

            model.fit(X_batch, y_batch)
            tr_prev = get_prev_from_labels(y_batch)

            if self.selection_metric == 'ds':
                self.train_distributions.append(getHist(posteriors[idx], 8))

            self.train_prevalences.append(tr_prev)
            self.models.append(model)

        self.sout('Fit [Done]')
        return self

    def predict(self, X):
        """Predict class prevalences by aggregating all ensemble members.

        Each fitted member produces a prevalence estimate; if a selection
        metric (``'ptr'`` or ``'ds'``) was configured, only the most relevant
        members are retained before computing the final ``mean`` or ``median``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        prevalences : dict or ndarray of shape (n_classes,)
            Estimated class prevalences, aggregated across the selected
            ensemble members.

        Examples
        --------
        >>> from mlquantify.meta import EnsembleQ
        >>> from mlquantify.matching import DyS
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=300, random_state=42)
        >>> q = EnsembleQ(DyS(estimator=LogisticRegression()), size=10).fit(X, y)
        >>> q.predict(X)
        {0: 0.49, 1: 0.51}
        """
        self.sout('Predict')

        test_prevalences = []

        for model in tqdm(self.models, disable=not self.verbose):
            with config_context(prevalence_return_type="array"):
                pred = np.asarray(model.predict(X))
            # Align predictions to self.classes so every model returns the
            # same number of entries even if its training subsample missed
            # some classes (defaulting missing classes to 0.0 prevalence).
            if len(pred) < len(self.classes) and hasattr(model, 'classes_'):
                aligned = np.zeros(len(self.classes))
                for i, c in enumerate(model.classes_):
                    idx = np.searchsorted(self.classes, c)
                    if idx < len(self.classes) and self.classes[idx] == c:
                        aligned[idx] = pred[i]
                pred = aligned
            test_prevalences.append(pred)

        test_prevalences = np.asarray(test_prevalences)

        if self.selection_metric == 'ptr':
            test_prevalences = self.ptr_selection_metric(test_prevalences, self.train_prevalences)
        elif self.selection_metric == 'ds':
            test_prevalences = self.ds_selection_metric(X,
                                                   test_prevalences,
                                                   self.train_distributions,
                                                   self.posteriors_generator)

        if self.return_type == "median":
            prevalences = np.median(test_prevalences, axis=0)
        else:
            prevalences = np.mean(test_prevalences, axis=0)


        self.sout('Predict [Done]')
        prevalences = validate_prevalences(self, prevalences, self.classes)
        return prevalences


    def ptr_selection_metric(self, prevalences, train_prevalences):
        r"""Select members whose training prevalence is closest to the test estimate.

        Computes an initial test-prevalence estimate by averaging all member
        predictions, then retains the top ``p_metric`` fraction of members
        ranked by how closely their training prevalence matches that estimate.

        Parameters
        ----------
        prevalences : ndarray of shape (n_members, n_classes)
            Prevalence estimates from each ensemble member.
        train_prevalences : list of dict or ndarray
            Training prevalences recorded for each ensemble member during
            ``fit``.

        Returns
        -------
        selected : list of ndarray
            Prevalence estimates from the selected subset of members.
        """
        test_prev_estim = prevalences.mean(axis=0)
        ptr_differences = [MSE(test_prev_estim, ptr_i) for ptr_i in train_prevalences]
        order = np.argsort(ptr_differences)
        return _select_k(prevalences, order, k=self.p_metric)

    def ds_get_posteriors(self, X, y):
        r"""Compute cross-validated posterior probabilities for the DS selection metric.

        Fits a logistic regression classifier with hyperparameters tuned by
        grid-search and returns out-of-fold posterior probabilities for the
        training data together with a callable for new instances.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        posteriors : ndarray of shape (n_samples, n_classes)
            Out-of-fold posterior probabilities for the training data.
        posteriors_generator : callable
            ``predict_proba`` method of the best-fitted estimator, used to
            generate posteriors for unseen test data during ``predict``.

        Notes
        -----
        Cross-validated posteriors ensure that no training sample is scored
        by a model trained on it, preventing over-optimistic score distributions.
        A separate logistic regression is used regardless of the base quantifier
        so that soft scores are always available for the DS metric.
        """
        lr_base = LogisticRegression(class_weight='balanced', max_iter=1000)

        optim = GridSearchCV(
            lr_base, param_grid={'C': np.logspace(-4, 4, 9)}, cv=5, n_jobs=self.n_jobs, refit=True
        ).fit(X, y)

        posteriors = cross_val_predict(
            optim.best_estimator_, X, y, cv=5, n_jobs=self.n_jobs, method='predict_proba'
        )
        posteriors_generator = optim.best_estimator_.predict_proba

        return posteriors, posteriors_generator


    def ds_selection_metric(self, X, prevalences, train_distributions, posteriors_generator):
        r"""Select members whose training score distribution is closest to the test distribution.

        Computes posterior-probability histograms for the test data and
        retains the top ``p_metric`` fraction of ensemble members ranked by
        Hellinger distance between their stored training histogram and the
        test histogram.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix used to compute the test score distribution.
        prevalences : ndarray of shape (n_members, n_classes)
            Prevalence estimates from each ensemble member.
        train_distributions : list of ndarray
            Posterior-probability histograms stored for each member during
            ``fit``.
        posteriors_generator : callable
            Function that returns posterior probabilities for new data
            (obtained from :meth:`ds_get_posteriors` during ``fit``).

        Returns
        -------
        selected : list of ndarray
            Prevalence estimates from the selected subset of members.
        """
        test_posteriors = posteriors_generator(X)
        test_distribution = getHist(test_posteriors, 8)
        dist = [hellinger(tr_dist_i, test_distribution) for tr_dist_i in train_distributions]
        order = np.argsort(dist)
        return _select_k(prevalences, order, k=self.p_metric)

def _select_k(elements, order, k):
    r"""
    Selects the k elements from the list of elements based on the order.
    If the list is empty, it returns the original list.

    Parameters
    ----------
    elements : array-like
        The array of elements to be selected from.
    order : array-like
        The order of the elements.
    k : int
        The number of elements to be selected.

    Returns
    -------
    array-like
        The selected elements.
    """
    if isinstance(k, float):
        k = max(1, int(k * len(elements)))
    elements_k = [elements[idx] for idx in order[:k]]
    if elements_k:
        return elements_k
    print(f"Unable to take {k} for elements with size {len(elements)}")
    return elements





class AggregativeBootstrap(MetaquantifierMixin, BaseQuantifier):
    r"""Aggregative Bootstrap quantifier for prevalence confidence regions.

    Targets prior probability shift. Wraps any aggregative quantifier and
    bootstrap-resamples its (cached) train and test predictions to turn a point
    prevalence estimate into a confidence region. Because aggregative
    quantifiers classify once and aggregate, the resampling is applied only to
    the cheap aggregation step, so uncertainty is obtained efficiently.

    Parameters
    ----------
    quantifier : BaseQuantifier
        The base aggregative quantifier to wrap.
    n_train_bootstraps : int, default=1
        Number of bootstrap resamples from the training predictions.
    n_test_bootstraps : int, default=1
        Number of bootstrap resamples from the test predictions; combining both
        sides captures model and sampling uncertainty.
    random_state : int or None, default=None
        Random seed for reproducibility.
    region_type : {'intervals', 'ellipse', 'ellipse-clr'}, default='intervals'
        Shape of the confidence region built from the resampled estimates.

        - ``'intervals'`` : independent per-class percentile intervals.
        - ``'ellipse'`` : Gaussian confidence ellipse on the simplex.
        - ``'ellipse-clr'`` : ellipse in centered-log-ratio (Aitchison) space,
          respecting the simplex geometry.
    confidence_level : float, default=0.95
        Probability mass the region is meant to cover.

    Attributes
    ----------
    train_predictions : ndarray
        Predictions on the training (or validation) set.
    y_train : ndarray
        Labels corresponding to ``train_predictions``.
    classes : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    Applying the bootstrap only to the aggregation step (not the whole pipeline)
    gives large speed-ups; the region is valid insofar as the bootstrap
    approximates the true sampling distribution, so it needs enough test points
    and resamples.

    See Also
    --------
    EnsembleQ : Ensemble wrapper for shift robustness.
    ConfidenceInterval : Confidence-region constructs produced here.

    Examples
    --------
    >>> from mlquantify.meta import AggregativeBootstrap
    >>> from mlquantify.likelihood import EMQ
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = AggregativeBootstrap(
    ...     EMQ(LogisticRegression()),
    ...     n_train_bootstraps=10,
    ...     n_test_bootstraps=10,
    ... ).fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}

    References
    ----------
    .. dropdown:: References

        .. [1] Moreo, A., & Salvati, A. (2025).
               Uncertainty Quantification in Quantification.
               *LQ 2025 Workshop Proceedings*.
    """

    _parameter_constraints = {
        "quantifier": [BaseQuantifier],
        "n_train_bootstraps": [Interval(left=1, right=None, discrete=True)],
        "n_test_bootstraps": [Interval(left=1, right=None, discrete=True)],
        "random_state": [Options([None, int])],
        "region_type": [Options(['intervals', 'ellipse', 'ellipse-clr'])],
        "confidence_level": [Interval(left=0.0, right=1.0)],
    }

    def __init__(self,
                 quantifier,
                 n_train_bootstraps=1,
                 n_test_bootstraps=1,
                 random_state=None,
                 region_type='intervals',
                 confidence_level=0.95):
        self.quantifier = quantifier
        self.n_train_bootstraps = n_train_bootstraps
        self.n_test_bootstraps = n_test_bootstraps
        self.random_state = random_state
        self.region_type = region_type
        self.confidence_level = confidence_level

    def fit(self, X, y, val_split=None):
        r"""Fit the base classifier and store predictions for bootstrap resampling.

        Trains only the base classifier (not the full aggregative quantifier),
        then stores the resulting soft predictions for later use in
        :meth:`aggregate`. Optionally holds out a validation split so the
        stored predictions come from unseen data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training class labels.
        val_split : float or None, default=None
            If given, the fraction of data held out as a validation set whose
            predictions are stored. ``None`` uses the full training set.

        Returns
        -------
        self : AggregativeBootstrap
            The fitted quantifier.

        Raises
        ------
        ValueError
            If the wrapped quantifier is not an aggregative quantifier.

        Examples
        --------
        >>> from mlquantify.meta import AggregativeBootstrap
        >>> from mlquantify.likelihood import EMQ
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = AggregativeBootstrap(EMQ(LogisticRegression()),
        ...                          n_train_bootstraps=10,
        ...                          n_test_bootstraps=10).fit(X, y)
        """
        X, y = validate_data(self, X, y)
        self.classes = np.unique(y)

        if not is_aggregative_quantifier(self.quantifier):
            raise ValueError(f"The quantifier {self.quantifier.__class__.__name__} is not an aggregative quantifier.")
        self.quantifier_estimator = deepcopy(self.quantifier)

        estimator_function = _get_estimator_function(self.quantifier_estimator)
        model = self.quantifier_estimator.estimator

        if val_split is None:
            model.fit(X, y)
            y_train = y
            train_predictions = getattr(model, estimator_function)(X)
        else:
            X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=val_split, random_state=self.random_state)
            model.fit(X_fit, y_fit)
            y_train = y_val
            train_predictions = getattr(model, estimator_function)(X_val)
        self.train_predictions = train_predictions
        self.y_train = y_train

        return self

    def predict(self, X):
        r"""Predict class prevalences with bootstrap-derived confidence estimation.

        Generates classifier predictions for ``X`` and delegates to
        :meth:`aggregate` with the stored training predictions to produce a
        bootstrap-based prevalence estimate.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        prevalences : dict or ndarray of shape (n_classes,)
            Point prevalence estimate extracted from the bootstrap distribution.

        Examples
        --------
        >>> from mlquantify.meta import AggregativeBootstrap
        >>> from mlquantify.likelihood import EMQ
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = AggregativeBootstrap(EMQ(LogisticRegression()),
        ...                          n_train_bootstraps=10,
        ...                          n_test_bootstraps=10).fit(X, y)
        >>> q.predict(X)
        {0: 0.49, 1: 0.51}
        """
        X = validate_data(self, X, None)
        estimator_function = _get_estimator_function(self.quantifier_estimator)
        model = self.quantifier_estimator.estimator

        predictions = getattr(model, estimator_function)(X)

        return self.aggregate(predictions, self.train_predictions, self.y_train)


    def aggregate(self, predictions, train_predictions, y_train):
        r"""Aggregate predictions via bootstrap resampling into a prevalence estimate.

        Resamples both the training and test predictions
        ``n_train_bootstraps × n_test_bootstraps`` times, calls the base
        quantifier's ``aggregate`` method on each combination, and summarises
        the resulting distribution as a point estimate with a confidence region.

        Parameters
        ----------
        predictions : ndarray of shape (n_test_samples, n_classes)
            Soft predictions on the test set (e.g. posterior probabilities).
        train_predictions : ndarray of shape (n_train_samples, n_classes)
            Soft predictions stored from the training (or validation) set.
        y_train : ndarray of shape (n_train_samples,)
            Class labels corresponding to ``train_predictions``.

        Returns
        -------
        prevalences : dict or ndarray of shape (n_classes,)
            Point prevalence estimate extracted from the centre of the
            bootstrap confidence region.

        Examples
        --------
        >>> from mlquantify.meta import AggregativeBootstrap
        >>> from mlquantify.likelihood import EMQ
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> import numpy as np
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> lr = LogisticRegression().fit(X, y)
        >>> train_preds = lr.predict_proba(X)
        >>> q = AggregativeBootstrap(EMQ(lr), n_train_bootstraps=5,
        ...                          n_test_bootstraps=5).fit(X, y)
        >>> q.aggregate(train_preds, train_preds, y)
        {0: 0.49, 1: 0.51}
        """
        prevalences = []

        self.classes = np.unique(y_train)

        for train_idx in bootstrap_sample_indices(
            n_samples=len(train_predictions),
            n_bootstraps=self.n_train_bootstraps,
            batch_size=len(train_predictions),
            random_state=self.random_state
        ):
            train_pred_boot = train_predictions[train_idx]
            train_y_boot = y_train[train_idx]

            for test_idx in bootstrap_sample_indices(
                n_samples=len(predictions),
                n_bootstraps=self.n_test_bootstraps,
                batch_size=len(predictions),
                random_state=self.random_state
            ):
                test_pred_boot = predictions[test_idx]

                requirements = get_aggregation_requirements(self.quantifier)

                with config_context(prevalence_return_type="array"):
                    if requirements.requires_train_proba and requirements.requires_train_labels:
                        prevalences_boot = self.quantifier.aggregate(test_pred_boot, train_pred_boot, train_y_boot)
                    elif requirements.requires_train_labels:
                        prevalences_boot = self.quantifier.aggregate(test_pred_boot, train_y_boot)
                    else:
                        prevalences_boot = self.quantifier.aggregate(test_pred_boot)
                prevalences.append(np.asarray(prevalences_boot))

        prevalences = np.asarray(prevalences)
        confidence_region = construct_confidence_region(
            prev_estims=prevalences,
            method=self.region_type,
            confidence_level=self.confidence_level,
        )

        prevalence = confidence_region.get_point_estimate()

        prevalence = validate_prevalences(self, prevalence, self.classes)

        return prevalence



@binary_quantifier(strategy_attr="strategy")
class QuaDapt(MetaquantifierMixin, BaseQuantifier):
    r"""QuaDapt: drift-resilient quantification via parameter adaptation.

    Targets general distribution shift / concept drift, not only prior shift.
    Wraps a soft base quantifier and, at prediction time, simulates training
    score distributions with MoSS at several overlap levels, selects the level
    whose mixed scores best match the test scores, and uses that synthetic set
    as the aggregation reference. Binary base method; multiclass via
    one-vs-rest.

    Parameters
    ----------
    quantifier : BaseQuantifier
        A soft (probabilistic) base aggregative quantifier.
    measure : {'topsoe', 'hellinger', 'probsymm', 'sord'}, default='topsoe'
        Distance comparing each synthetic score set with the test scores, used
        to pick the best merging factor.

        - ``'topsoe'`` : symmetric information-theoretic distance.
        - ``'hellinger'`` : bounded sqrt-probability distance.
        - ``'probsymm'`` : probabilistic symmetric chi-square distance.
        - ``'sord'`` : Sample Ordinal Distance on the raw scores (bin-free).
    merging_factors : array-like, default=np.arange(0.1, 1.0, 0.2)
        Candidate MoSS overlap levels ``m`` to evaluate; each sets how much the
        synthetic positive and negative scores overlap (0 = well separated,
        1 = fully overlapping).
    strategy : {'ovr', 'ovo'}, default='ovr'
        Multiclass decomposition strategy.

        - ``'ovr'`` : one-vs-rest, one binary adaptation per class.
        - ``'ovo'`` : one-vs-one, one binary adaptation per class pair.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    y_train : ndarray of shape (n_samples,)
        Training labels stored during ``fit``.

    Notes
    -----
    Built on MoSS (Model for Score Simulation): adapting the merging factor to
    the test scores makes a standard quantifier resilient when the
    score-distribution complexity drifts. Generalises DySyn (DyS + MoSS) to any
    classifier-based quantifier; only the quantifier's score reference is
    adapted, with no classifier retraining.

    See Also
    --------
    DyS : A common base quantifier for QuaDapt.
    EnsembleQ : Ensemble wrapper for prior-shift robustness.

    Examples
    --------
    >>> from mlquantify.meta import QuaDapt
    >>> from mlquantify.matching import DyS
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = QuaDapt(DyS(LogisticRegression())).fit(X, y)
    >>> q.predict(X)
    {0: 0.49, 1: 0.51}
    >>> # call aggregate with pre-computed posteriors
    >>> proba = LogisticRegression().fit(X, y).predict_proba(X)
    >>> q.aggregate(proba, y)
    {0: 0.49, 1: 0.51}

    References
    ----------
    .. dropdown:: References

        .. [1] Ortega, J. P., Luth Junior, L. F., Zalewski, W., & Maletzke, A.
               (2025). QuaDapt: Drift-Resilient Quantification via Parameters
               Adaptation. *Proc. 5th Int. Workshop on Learning to Quantify
               (LQ 2025)*, p. 64.
        .. [2] Maletzke, A., dos Reis, D., Hassan, W., & Batista, G. (2021).
               Accurately Quantifying under Score Variability. *ICDM 2021*,
               pp. 1228–1233. (Model for Score Simulation, MoSS.)
    """

    _parameter_constraints = {
        "quantifier": [BaseQuantifier],
        "merging_factors": ["array-like"],
        "measure": [Options(["hellinger", "topsoe", "probsymm", "sord"])],
        "strategy": [Options(["ovr", "ovo"])]
    }

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        tags.prediction_requirements.requires_train_labels = True
        return tags

    def __init__(self,
                 quantifier,
                 measure="topsoe",
                 merging_factors=np.arange(0.1, 1.0, 0.2),
                 strategy="ovr"):
        self.quantifier = quantifier
        self.measure = measure
        self.merging_factors = merging_factors
        self.strategy = strategy


    def fit(self, X, y):
        """Fit the base classifier of the wrapped quantifier.

        Only the underlying estimator is trained here; the full aggregation
        is deferred to :meth:`aggregate` so that the MoSS-based correction
        can be applied at prediction time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : QuaDapt
            The fitted quantifier.

        Raises
        ------
        ValueError
            If the wrapped quantifier does not use soft (probabilistic)
            predictions.

        Examples
        --------
        >>> from mlquantify.meta import QuaDapt
        >>> from mlquantify.matching import DyS
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = QuaDapt(DyS(LogisticRegression())).fit(X, y)
        """
        X, y = validate_data(self, X, y)
        self.classes_ = np.unique(y)

        if not uses_soft_predictions(self.quantifier):
            raise ValueError(f"The quantifier {self.quantifier.__class__.__name__} is not a soft (probabilistic) quantifier.")

        self.quantifier.estimator.fit(X, y)
        self.y_train = y

        return self

    def predict(self, X):
        """Predict class prevalences using the MoSS adaptive correction.

        Generates posterior probabilities for ``X`` with the fitted classifier
        and delegates to :meth:`aggregate`, which selects the best MoSS
        merging factor and calls the base quantifier's ``aggregate``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        prevalences : dict or ndarray of shape (n_classes,)
            Estimated class prevalences.

        Examples
        --------
        >>> from mlquantify.meta import QuaDapt
        >>> from mlquantify.matching import DyS
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = QuaDapt(DyS(LogisticRegression())).fit(X, y)
        >>> q.predict(X)
        {0: 0.49, 1: 0.51}
        """
        X = validate_data(self, X, None)

        model = self.quantifier.estimator

        predictions = getattr(model, "predict_proba")(X)

        return self.aggregate(predictions, self.y_train)


    def aggregate(self, predictions, y_train):
        """Aggregate posteriors into prevalences using MoSS score simulation.

        Searches over ``merging_factors`` to find the synthetic score
        distribution (generated by :meth:`MoSS`) whose histogram is closest
        to the test score distribution, then passes that synthetic set as the
        training reference to the base quantifier's ``aggregate``.

        Parameters
        ----------
        predictions : ndarray of shape (n_samples, n_classes)
            Posterior probabilities of the test instances.
        y_train : ndarray of shape (n_train_samples,)
            Training class labels used to resolve class ordering.

        Returns
        -------
        prevalences : dict or ndarray of shape (n_classes,)
            Estimated class prevalences.

        Examples
        --------
        >>> from mlquantify.meta import QuaDapt
        >>> from mlquantify.matching import DyS
        >>> from sklearn.linear_model import LogisticRegression
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, random_state=42)
        >>> q = QuaDapt(DyS(LogisticRegression())).fit(X, y)
        >>> proba = LogisticRegression().fit(X, y).predict_proba(X)
        >>> q.aggregate(proba, y)
        {0: 0.49, 1: 0.51}
        """
        self.classes_ = check_classes_attribute(self, np.unique(y_train))
        _, _, best_m = self.best_mixture(predictions)

        moss_scores, moss_labels = self.MoSS(n=1000, alpha=0.5, merging_factor=best_m, classes=self.classes_)

        prevalences = self.quantifier.aggregate(predictions, moss_scores, moss_labels)

        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences


    def best_mixture(self, predictions):
        """Find the merging factor and prevalence that best match the test scores.

        Evaluates each candidate value in ``merging_factors`` by generating a
        synthetic score set with :meth:`MoSS` and measuring its distance to
        the test distribution using the configured ``measure``. Returns the
        merging factor, prevalence estimate, and distance for the best match.

        Parameters
        ----------
        predictions : ndarray of shape (n_samples,) or (n_samples, 2)
            Posterior probabilities or positive-class scores for the test set.

        Returns
        -------
        best_alpha : float
            Positive-class prevalence estimate under the best merging factor.
        best_distance : float
            Distance between the test distribution and the best synthetic mix.
        best_m : float
            Merging factor that achieved the lowest distance.
        """
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim == 2:
            predictions = predictions[:, 1]
        else:
            predictions = predictions.ravel()

        MF = np.atleast_1d(np.round(self.merging_factors, 2)).astype(float)

        distances = []
        alphas = []

        for mf in MF:
            scores, labels = self.MoSS(n=1000, alpha=0.5, merging_factor=mf)
            pos_scores = scores[labels == self.classes_[1]][:, 1]
            neg_scores = scores[labels == self.classes_[0]][:, 1]

            if self.measure in ["hellinger", "topsoe", "probsymm"]:
                alpha, distance = self._histogram_best_mixture(
                    predictions,
                    pos_scores,
                    neg_scores,
                    self.measure,
                )
            elif self.measure == "sord":
                alpha, distance = self._sord_best_mixture(
                    predictions,
                    pos_scores,
                    neg_scores,
                )

            distances.append(distance)
            alphas.append(alpha)

        best_m = MF[np.argmin(distances)]
        best_alpha = alphas[np.argmin(distances)]
        best_distance = np.min(distances)
        return best_alpha, best_distance, best_m

    def _histogram_best_mixture(self, predictions, pos_scores, neg_scores, distance):
        bins_size = getattr(
            self.quantifier,
            "bins_size",
            np.append(np.linspace(2, 20, 10), 30).astype(int),
        )
        loss_function = get_loss(loss=distance, normalize=True)

        best_alpha = 0.0
        best_distance = np.inf

        for n_bins in np.atleast_1d(bins_size).astype(int):
            test_hist = get_histogram(predictions, n_bins)
            pos_hist = get_histogram(pos_scores, n_bins)
            neg_hist = get_histogram(neg_scores, n_bins)

            def objective(alpha):
                mixture = (1.0 - alpha) * neg_hist + alpha * pos_hist
                return loss_function(mixture, test_hist)

            prevalences, current_distance = minimize_prevalence(
                objective=objective,
                n_classes=2,
                solver="grid",
            )

            if current_distance < best_distance:
                best_alpha = float(prevalences[1])
                best_distance = float(current_distance)

        return best_alpha, best_distance

    @staticmethod
    def _sord_best_mixture(predictions, pos_scores, neg_scores):
        predictions = np.asarray(predictions, dtype=float).ravel()
        pos_scores = np.asarray(pos_scores, dtype=float).ravel()
        neg_scores = np.asarray(neg_scores, dtype=float).ravel()

        scores = np.concatenate([pos_scores, neg_scores, predictions])
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        gaps = np.diff(sorted_scores)

        n_pos = len(pos_scores)
        n_neg = len(neg_scores)
        n_test = len(predictions)

        def objective(alpha):
            weights = np.concatenate(
                [
                    np.full(n_pos, alpha / n_pos),
                    np.full(n_neg, (1.0 - alpha) / n_neg),
                    np.full(n_test, -1.0 / n_test),
                ]
            )
            sorted_weights = weights[order]
            cumulative_weights = np.cumsum(sorted_weights)[:-1]
            return float(np.sum(np.abs(gaps * cumulative_weights)))

        prevalences, distance = minimize_prevalence(
            objective=objective,
            n_classes=2,
            solver="grid",
        )

        return float(prevalences[1]), float(distance)

    def get_best_distance(self, predictions):
        """Return the minimum distribution distance achieved across all merging factors.

        Parameters
        ----------
        predictions : ndarray of shape (n_samples,) or (n_samples, 2)
            Posterior probabilities or positive-class scores for the test set.

        Returns
        -------
        best_distance : float
            Lowest distance between the test distribution and any synthetic mix.
        """
        _, distance, _ = self.best_mixture(predictions)

        return distance


    @classmethod
    def MoSS(cls, n, alpha, merging_factor, classes=None, random_state=None):
        r"""Generate a synthetic binary score set via the Model for Score Simulation.

        Positive scores are sampled as :math:`U^{\mathfrak{m}}` and negative
        scores as :math:`1 - U^{\mathfrak{m}}`, where :math:`U \sim
        \mathrm{Uniform}(0,1)` and :math:`\mathfrak{m}` is the merging factor.
        A higher ``merging_factor`` produces more overlapping positive and
        negative score distributions.

        Parameters
        ----------
        n : int
            Total number of synthetic observations to generate.
        alpha : float
            Prevalence of the positive class in the synthetic set.
        merging_factor : float
            Controls the overlap between positive and negative score
            distributions. Values close to 0 produce well-separated scores;
            values close to 1 produce heavily overlapping scores.
        classes : array-like of length 2 or None, default=None
            Class labels for the negative and positive class respectively.
            If ``None``, labels ``0`` and ``1`` are used.
        random_state : int or None, default=None
            Unused; reserved for future reproducibility support.

        Returns
        -------
        scores : ndarray of shape (n, 2)
            Synthetic soft predictions for each observation.
        labels : ndarray of shape (n,)
            Class labels for each synthetic observation.

        .. math::

            \mathrm{MoSS}(n, \alpha, \mathfrak{m}) =
            \mathrm{syn}(\oplus, \lfloor \alpha n \rfloor, \mathfrak{m})
            \cup
            \mathrm{syn}(\ominus, \lfloor (1-\alpha) n \rfloor, \mathfrak{m})

        Notes
        -----
        Only binary score generation is supported. The method is used
        internally by :meth:`aggregate` to build a synthetic training
        reference for the base quantifier.

        Examples
        --------
        >>> from mlquantify.meta import QuaDapt
        >>> scores, labels = QuaDapt.MoSS(n=1000, alpha=0.3, merging_factor=0.5)
        >>> scores.shape
        (1000, 2)
        >>> labels.shape
        (1000,)

        References
        ----------
        .. dropdown:: References

            .. [1] Maletzke, A., Reis, D., Hassan, W., & Batista, G. (2021).
                   Accurately Quantifying under Score Variability.
                   *ICDM 2021*, pp. 1228–1233.
        """
        if isinstance(alpha, list):
            alpha = float(alpha[1])

        # Define os rótulos das classes
        if classes is None:
            neg_label, pos_label = 0, 1
        else:
            if len(classes) < 2:
                raise ValueError("classes must contain exactly two elements.")

            neg_label, pos_label = classes[0], classes[1]

        n_pos = int(n * alpha)
        n_neg = n - n_pos

        # Scores positivos
        p_score = np.random.uniform(size=n_pos) ** merging_factor
        # Scores negativos
        n_score = 1 - (np.random.uniform(size=n_neg) ** merging_factor)

        # Labels
        pos_labels = np.full(n_pos, pos_label)
        neg_labels = np.full(n_neg, neg_label)

        moss = np.column_stack(
            (
                1 - np.concatenate((p_score, n_score)),
                np.concatenate((p_score, n_score)),
                np.concatenate((pos_labels, neg_labels)),
            )
        )

        scores = moss[:, :2]
        labels = moss[:, 2].astype(type(pos_label))
        return scores, labels
