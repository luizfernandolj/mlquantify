import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import (
    AggregativeMixin,
    CrispPredictionMixin,
)
from mlquantify.counting import CC
from mlquantify.tree._classification import QuantificationTreeClassifier
from mlquantify.utils._decorators import _fit_context
from mlquantify.utils._get_scores import apply_cross_validation
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._validation import (
    validate_data,
    validate_prevalences,
    resolve_aggregate_classes,
)


def _fit_forest_tree(tree, X, y, classes, adjusted, cv, seed):
    """Fit one forest member; optionally estimate its one-vs-rest tpr/fpr
    rates via cross-validation on its own training records (Milli et al.,
    2013, use 10-fold CV for the Adjusted Count correction)."""
    n_classes = len(classes)
    tpr = np.ones(n_classes)
    fpr = np.zeros(n_classes)

    if not adjusted:
        return tree.fit(X, y), tpr, fpr

    class_counts = np.array([np.count_nonzero(y == _class) for _class in classes])
    n_folds = int(min(cv, class_counts[class_counts > 0].min()))
    if n_folds < 2:
        return tree.fit(X, y), tpr, fpr

    oof_predictions, y_oof, fitted_tree = apply_cross_validation(
        tree, X, y,
        cv=n_folds,
        function="predict",
        stratified=True,
        shuffle=True,
        random_state=seed,
        cv_prediction="refit",
    )
    for idx, _class in enumerate(classes):
        positive = y_oof == _class
        if positive.any():
            tpr[idx] = np.mean(oof_predictions[positive] == _class)
        if (~positive).any():
            fpr[idx] = np.mean(oof_predictions[~positive] == _class)
    return fitted_tree, tpr, fpr


_TREE_PARAMETER_CONSTRAINTS = {
    "criterion": [Options(["eb", "cqb"])],
    "max_depth": [Interval(1, None, inclusive_right=False), type(None)],
    "min_samples_split": [Interval(2, None, inclusive_right=False)],
    "min_samples_leaf": [Interval(1, None, inclusive_right=False)],
    "max_features": [
        Options(["sqrt", "log2"]),
        Interval(0.0, 1.0),
        Interval(1, None, inclusive_right=False),
        type(None),
    ],
    "random_state": [Interval(0, None, inclusive_right=False), type(None)],
}


class QuantificationTree(CC):
    r"""Quantification Tree (QTree) quantifier.

    Targets prior probability shift. QuantificationTree is an **aggregative**
    Classify-and-Count quantifier — it shares the standard ``fit`` /
    ``predict`` / ``aggregate`` interface of :class:`~mlquantify.counting.CC`
    — but its classifier is a decision tree *grown for quantification*
    (:class:`QuantificationTreeClassifier`): splits are chosen to balance
    false positives against false negatives per class, so that plain counting
    of the leaf predictions estimates the class prevalences directly. Like
    :class:`~mlquantify.neighbors.PWK`, it therefore takes **no external
    estimator parameter**: the quantification tree is intrinsic to the method.

    Parameters
    ----------
    criterion : {'eb', 'cqb'}, default='cqb'
        Split-quality measure of the underlying tree:

        - ``'eb'`` (Classification Error Balancing):
          :math:`QE[c] = |FP_c - FN_c|`, optimising quantification error only.
        - ``'cqb'`` (Classification-Quantification Balancing):
          :math:`QE[c] = |FP_c^2 - FN_c^2|`, trading off classification and
          quantification error.
    max_depth : int or None, default=None
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum number of samples required to attempt a split.
    min_samples_leaf : int, default=1
        Minimum number of samples required in each child.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Number of features examined at each split (see
        :class:`QuantificationTreeClassifier`).
    random_state : int or None, default=None
        Seed for the per-split feature subsampling.

    Attributes
    ----------
    estimator : QuantificationTreeClassifier
        The underlying quantification tree (built from the parameters above;
        not an argument).
    estimator_ : QuantificationTreeClassifier
        The fitted tree.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Notes
    -----
    A split is accepted only when it strictly decreases the L2 norm of the
    per-class quantification error vector of the whole tree over the training
    set; when no split has positive gain the node becomes a leaf labelled with
    its majority class. The paper also evaluates Adjusted-Count
    post-processing on top of the tree, which in mlquantify is obtained by
    composition, e.g. ``ACC(estimator=QuantificationTreeClassifier())``.

    See Also
    --------
    QuantificationTreeClassifier : The underlying tree learner.
    QuantificationForest : Prevalence-averaging forest of quantification trees.
    CC : Plain classify-and-count baseline.

    Examples
    --------
    >>> from mlquantify.tree import QuantificationTree
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = QuantificationTree(criterion='cqb', random_state=0).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Milli, L., Monreale, A., Rossetti, G., Giannotti, F., Pedreschi, D.,
               & Sebastiani, F. (2013). Quantification Trees. *IEEE ICDM*,
               pp. 528-536.
    """

    _parameter_constraints = dict(_TREE_PARAMETER_CONSTRAINTS)

    def __init__(self,
                 criterion="cqb",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features=None,
                 random_state=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        # The quantification tree is intrinsic to the method rather than
        # supplied by the user.
        super().__init__(estimator=self._make_tree())

    def _make_tree(self):
        return QuantificationTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
        )

    def fit(self, X, y, estimator_fitted=False, *args, **kwargs):
        r"""Fit the quantification tree on the provided data."""
        # Rebuild the intrinsic tree so parameters changed via ``set_params``
        # (e.g. during grid search) take effect.
        if not estimator_fitted:
            self.estimator = self._make_tree()
        return super().fit(X, y, estimator_fitted=estimator_fitted, *args, **kwargs)


class QuantificationForest(CrispPredictionMixin, AggregativeMixin, BaseQuantifier):
    r"""Random Forest quantifier over quantification trees.

    Targets prior probability shift. Builds ``n_estimators``
    :class:`QuantificationTreeClassifier` trees, each on a random fraction of
    the training records with a random subset of features, and estimates the
    test prevalences as the **average of the per-tree estimates**
    (wisdom-of-the-crowd aggregation of Milli et al., 2013; note it averages
    prevalences, not votes). Following the paper, each tree's estimate is by
    default the **Adjusted Count**
    :math:`p_c = (\hat{p}_c^{CC} - fpr_c) / (tpr_c - fpr_c)`, with the
    per-class rates estimated by cross-validation on the tree's training
    records; set ``adjusted=False`` for plain Classify-and-Count averaging.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of quantification trees in the forest.
    sample_fraction : float, default=1.0
        Fraction of the training records drawn (without replacement) for
        each tree.
    criterion : {'eb', 'cqb'}, default='cqb'
        Split-quality measure of the trees (see :class:`QuantificationTree`).
    max_depth : int or None, default=None
        Maximum depth of each tree.
    min_samples_split : int, default=2
        Minimum number of samples required to attempt a split.
    min_samples_leaf : int, default=1
        Minimum number of samples required in each child.
    max_features : int, float, {'sqrt', 'log2'} or None, default='log2'
        Number of features examined at each split. The default ``'log2'``
        (:math:`\lfloor\log_2 d\rfloor + 1` features) follows the original
        Random Forest quantifier.
    adjusted : bool, default=True
        Whether each tree applies the Adjusted Count correction (formula (1)
        of the paper) to its Classify-and-Count estimate before averaging.
    cv : int, default=10
        Number of cross-validation folds used to estimate each tree's
        ``tpr``/``fpr`` rates when ``adjusted=True`` (the paper uses 10).
    n_jobs : int or None, default=None
        Number of parallel jobs used to fit the trees.
    random_state : int or None, default=None
        Seed controlling record subsampling and per-tree feature subsampling.

    Attributes
    ----------
    estimator : QuantificationTreeClassifier
        Prototype tree built from the parameters above (not an argument).
    estimator_ : list of QuantificationTreeClassifier
        The fitted trees.
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    tpr_ : ndarray of shape (n_estimators, n_classes)
        Per-tree, per-class (one-vs-rest) true positive rates estimated by
        cross-validation. Only set when ``adjusted=True``.
    fpr_ : ndarray of shape (n_estimators, n_classes)
        Per-tree, per-class (one-vs-rest) false positive rates estimated by
        cross-validation. Only set when ``adjusted=True``.

    See Also
    --------
    QuantificationTree : Single quantification tree quantifier.
    QuantificationTreeClassifier : The underlying tree learner.

    Examples
    --------
    >>> from mlquantify.tree import QuantificationForest
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=200, random_state=42)
    >>> q = QuantificationForest(n_estimators=10, random_state=0).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Milli, L., Monreale, A., Rossetti, G., Giannotti, F., Pedreschi, D.,
               & Sebastiani, F. (2013). Quantification Trees. *IEEE ICDM*,
               pp. 528-536.
        .. [2] Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
    """

    _parameter_constraints = {
        **_TREE_PARAMETER_CONSTRAINTS,
        "n_estimators": [Interval(1, None, inclusive_right=False)],
        "sample_fraction": [Interval(0.0, 1.0, inclusive_left=False)],
        "adjusted": [bool],
        "cv": [Interval(2, None, inclusive_right=False)],
        "n_jobs": [Interval(-1, None, inclusive_right=False), type(None)],
    }

    def __init__(self,
                 n_estimators=100,
                 sample_fraction=1.0,
                 criterion="cqb",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features="log2",
                 adjusted=True,
                 cv=10,
                 n_jobs=None,
                 random_state=None):
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.adjusted = adjusted
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        # Prototype tree; ``fit`` builds one per member of the forest.
        self.estimator = QuantificationTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        )

    def __mlquantify_tags__(self):
        tags = super().__mlquantify_tags__()
        tags.prediction_requirements.requires_train_proba = False
        tags.prediction_requirements.requires_train_labels = False
        return tags

    def _make_tree(self, seed):
        return QuantificationTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=seed,
        )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        r"""Fit the forest of quantification trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        self : QuantificationForest
            Fitted quantifier.
        """
        X, y = validate_data(self, X, y)
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        n_samples = X.shape[0]
        subsample_size = max(1, int(round(self.sample_fraction * n_samples)))
        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=self.n_estimators)
        subsamples = [
            rng.choice(n_samples, size=subsample_size, replace=False)
            for _ in range(self.n_estimators)
        ]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_forest_tree)(
                self._make_tree(seed), X[subsample], y[subsample],
                self.classes_, self.adjusted, self.cv, seed,
            )
            for seed, subsample in zip(seeds, subsamples)
        )
        self.estimator_ = [tree for tree, _, _ in results]
        self.tpr_ = np.stack([tpr for _, tpr, _ in results])
        self.fpr_ = np.stack([fpr for _, _, fpr in results])
        return self

    def predict(self, X):
        r"""Predict class prevalences as the average of the per-tree estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        ndarray of shape (n_classes,)
            Class prevalence estimates.
        """
        X = validate_data(self, X)
        predictions = np.stack([tree.predict(X) for tree in self.estimator_])
        return self.aggregate(predictions)

    def aggregate(self, predictions, classes=None):
        r"""Aggregate per-tree label predictions into prevalence estimates.

        Each row of ``predictions`` is turned into a Classify-and-Count
        prevalence vector, corrected per tree with the Adjusted Count formula
        when the forest was fitted with ``adjusted=True``; the forest
        estimate is their mean.

        Parameters
        ----------
        predictions : ndarray of shape (n_trees, n_samples)
            Crisp label predictions of each tree on the test data. A single
            1D row of labels is also accepted. When called standalone (with
            predictions not produced by the fitted trees, or on an unfitted
            quantifier), no adjustment is applied.
        classes : array-like of shape (n_classes,) or None, default=None
            Class labels the output must report, in order. When ``None``, the
            classes seen during ``fit`` are used; if the quantifier is
            unfitted, they are inferred from the predictions.

        Returns
        -------
        ndarray of shape (n_classes,)
            Class prevalence estimates.

        Examples
        --------
        >>> from mlquantify.tree import QuantificationForest
        >>> import numpy as np
        >>> q = QuantificationForest()
        >>> predictions = np.random.randint(0, 2, size=(10, 200))
        >>> q.aggregate(predictions)
        {0: ..., 1: ...}
        """
        predictions = np.atleast_2d(np.asarray(predictions))
        self.classes_ = resolve_aggregate_classes(
            self, classes, getattr(self, "classes_", None), predictions.ravel()
        )
        per_tree = np.stack([
            np.array([np.count_nonzero(row == _class) for _class in self.classes_])
            / row.shape[0]
            for row in predictions
        ])

        tpr = getattr(self, "tpr_", None)
        fpr = getattr(self, "fpr_", None)
        if (
            tpr is not None
            and tpr.shape == per_tree.shape
            and fpr.shape == per_tree.shape
        ):
            denominator = tpr - fpr
            adjustable = np.abs(denominator) > 1e-12
            adjusted = np.where(
                adjustable,
                (per_tree - fpr) / np.where(adjustable, denominator, 1.0),
                per_tree,
            )
            per_tree = np.clip(adjusted, 0.0, 1.0)
            row_sums = per_tree.sum(axis=1, keepdims=True)
            per_tree = np.where(row_sums > 0, per_tree / np.where(row_sums > 0, row_sums, 1.0),
                                1.0 / per_tree.shape[1])

        prevalences = per_tree.mean(axis=0)
        prevalences = validate_prevalences(self, prevalences, self.classes_)
        return prevalences
