import numpy as np
from sklearn.utils import check_X_y, check_array, check_random_state


class _Node:
    """A single node of a fitted quantification tree.

    Internal nodes carry a ``feature``/``threshold`` binary test
    (``x[feature] <= threshold`` goes left); leaves carry the majority
    ``label`` index and the training class-frequency vector ``proba``.
    """

    __slots__ = ("feature", "threshold", "left", "right", "label", "proba")

    def __init__(self, label, proba):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.label = label
        self.proba = proba

    def is_leaf(self):
        return self.left is None


class QuantificationTreeClassifier:
    r"""Decision tree classifier optimized for quantification.

    A CART-like binary decision tree whose splits are chosen to balance
    false positives against false negatives per class, rather than to
    minimise impurity. Since Classify & Count is exact whenever
    :math:`FP_c = FN_c` for every class, the tree greedily minimises a
    tree-level quantification error computed from the leaf predictions
    over the whole training set (Milli et al., 2013).

    Parameters
    ----------
    criterion : {'eb', 'cqb'}, default='cqb'
        Measure of the per-class quantification error :math:`QE[c]`:

        - ``'eb'`` (Classification Error Balancing):
          :math:`QE[c] = |FP_c - FN_c|`.
        - ``'cqb'`` (Classification-Quantification Balancing):
          :math:`QE[c] = |FP_c^2 - FN_c^2| = |FP_c - FN_c|\,(FP_c + FN_c)`,
          trading off quantification against classification error.

        A split is accepted if it strictly decreases :math:`\|QE\|_2` over
        the training set (positive gain), or keeps it unchanged while
        strictly reducing the number of misclassified training samples.
    max_depth : int or None, default=None
        Maximum tree depth. ``None`` means unlimited.
    min_samples_split : int, default=2
        Minimum number of samples required to attempt a split.
    min_samples_leaf : int, default=1
        Minimum number of samples required in each child.
    max_features : int, float, {'sqrt', 'log2'} or None, default=None
        Number of features examined at each split. ``None`` uses all
        features; ``'sqrt'`` uses :math:`\lfloor\sqrt{d}\rfloor`;
        ``'log2'`` uses :math:`\lfloor\log_2 d\rfloor + 1` (the value used
        by the Random Forest quantifier in the original paper); a float is
        interpreted as a fraction of the features.
    random_state : int, RandomState or None, default=None
        Seed for the per-split feature subsampling.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    tree_ : _Node
        Root of the fitted tree.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Notes
    -----
    The gain of a candidate split is **global**: the per-class ``FP``/``FN``
    counts of the whole current tree are updated as if the node were
    replaced by the two candidate children (each predicting its majority
    class), and the split maximising the decrease of :math:`\|QE\|_2` is
    kept, breaking ties by the reduction in misclassified samples.

    The original paper stops as soon as no split has strictly positive
    gain. Because :math:`\|QE\|_2` frequently reaches zero after a single
    split (any split with exactly balanced errors), that rule alone
    degenerates into a decision stump; this implementation therefore also
    accepts zero-gain splits while they strictly reduce the
    misclassification count, so the tree keeps improving as a classifier
    without ever worsening its quantification error.

    Examples
    --------
    >>> import numpy as np
    >>> X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
    >>> clf = QuantificationTreeClassifier(criterion='eb').fit(X, y)
    >>> labels = clf.predict(X)
    >>> proba = clf.predict_proba(X)

    References
    ----------
    Milli, L., Monreale, A., Rossetti, G., Giannotti, F., Pedreschi, D., &
    Sebastiani, F. (2013). Quantification Trees. *IEEE ICDM*, pp. 528-536.
    """

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

    def get_params(self, deep=True):
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _qe(self, fp, fn):
        if self.criterion == "eb":
            return np.abs(fp - fn)
        return np.abs(fp - fn) * (fp + fn)

    def _resolve_max_features(self, n_features):
        max_features = self.max_features
        if max_features is None:
            return n_features
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            # log2(d) + 1 features, as in the Random Forest quantifier of
            # Milli et al. (2013).
            return min(n_features, max(1, int(np.log2(n_features)) + 1))
        if isinstance(max_features, float) and not isinstance(max_features, bool):
            return min(n_features, max(1, int(max_features * n_features)))
        return min(n_features, max(1, int(max_features)))

    def fit(self, X, y):
        """Build the quantification tree from the training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : QuantificationTreeClassifier
            The fitted classifier.
        """
        if self.criterion not in ("eb", "cqb"):
            raise ValueError(
                f"criterion must be 'eb' or 'cqb', got {self.criterion!r}."
            )
        X, y = check_X_y(X, y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        n_classes = len(self.classes_)
        rng = check_random_state(self.random_state)
        n_subset = self._resolve_max_features(self.n_features_in_)

        counts = np.bincount(y_idx, minlength=n_classes)
        label = int(np.argmax(counts))
        # Global FP/FN state of the tree-so-far (root as a leaf predicting
        # its majority class); every candidate split is scored against it.
        self._fp = np.zeros(n_classes, dtype=float)
        self._fn = counts.astype(float)
        self._fp[label] = len(y_idx) - counts[label]
        self._fn[label] = 0.0

        self.tree_ = self._build(X, y_idx, np.arange(len(y_idx)), counts,
                                 depth=0, n_subset=n_subset, rng=rng)
        del self._fp, self._fn
        return self

    def _build(self, X, y_idx, indices, counts, depth, n_subset, rng):
        n_classes = counts.shape[0]
        label = int(np.argmax(counts))
        node = _Node(label, counts / counts.sum())
        n_node = len(indices)

        if (
            n_node < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.count_nonzero(counts) <= 1
        ):
            return node

        split = self._find_best_split(
            X[indices], y_idx[indices], counts, label, n_subset, rng
        )
        if split is None:
            return node
        gain, error_after, feature, threshold = split
        # Accept a split when it strictly decreases the tree-level QE norm
        # (the paper's rule), or keeps it unchanged while strictly reducing
        # the misclassification count — without the tie-break the norm often
        # reaches zero after one split (FP = FN exactly) and the tree would
        # degenerate to a stump.
        error_before = self._fn.sum()
        if gain <= 1e-9 and not (gain > -1e-9 and error_after < error_before - 0.5):
            return node
        mask = X[indices, feature] <= threshold
        left_indices, right_indices = indices[mask], indices[~mask]
        counts_left = np.bincount(y_idx[left_indices], minlength=n_classes)
        counts_right = counts - counts_left

        # Commit the split: replace this node's contribution to the global
        # FP/FN state with the two children's.
        self._remove_leaf(counts, label)
        self._add_leaf(counts_left, int(np.argmax(counts_left)))
        self._add_leaf(counts_right, int(np.argmax(counts_right)))

        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X, y_idx, left_indices, counts_left,
                                depth + 1, n_subset, rng)
        node.right = self._build(X, y_idx, right_indices, counts_right,
                                 depth + 1, n_subset, rng)
        return node

    def _add_leaf(self, counts, label):
        self._fp[label] += counts.sum() - counts[label]
        self._fn += counts
        self._fn[label] -= counts[label]

    def _remove_leaf(self, counts, label):
        self._fp[label] -= counts.sum() - counts[label]
        self._fn -= counts
        self._fn[label] += counts[label]

    def _find_best_split(self, X_node, y_node, counts, label, n_subset, rng):
        n_node, n_features = X_node.shape
        n_classes = counts.shape[0]

        # FP/FN of the rest of the tree, with this node's leaf removed.
        base_fp, base_fn = self._fp.copy(), self._fn.copy()
        base_fp[label] -= n_node - counts[label]
        base_fn -= counts
        base_fn[label] += counts[label]

        parent_score = np.linalg.norm(self._qe(self._fp, self._fn))
        onehot = np.eye(n_classes)[y_node]

        if n_subset < n_features:
            features = rng.choice(n_features, size=n_subset, replace=False)
        else:
            features = np.arange(n_features)

        best = None
        best_key = None
        sizes_left = np.arange(1, n_node)
        size_ok = (sizes_left >= self.min_samples_leaf) & \
                  (n_node - sizes_left >= self.min_samples_leaf)

        for feature in features:
            values = X_node[:, feature]
            order = np.argsort(values, kind="mergesort")
            values_sorted = values[order]
            valid = size_ok & (values_sorted[1:] > values_sorted[:-1])
            if not valid.any():
                continue

            counts_left = np.cumsum(onehot[order], axis=0)[:-1][valid]
            counts_right = counts[None, :] - counts_left
            thresholds = (values_sorted[:-1][valid] + values_sorted[1:][valid]) / 2

            n_thresholds = counts_left.shape[0]
            rows = np.arange(n_thresholds)
            pred_left = counts_left.argmax(axis=1)
            pred_right = counts_right.argmax(axis=1)

            new_fp = np.tile(base_fp, (n_thresholds, 1))
            new_fp[rows, pred_left] += counts_left.sum(axis=1) - counts_left[rows, pred_left]
            new_fp[rows, pred_right] += counts_right.sum(axis=1) - counts_right[rows, pred_right]

            fn_left = counts_left.copy()
            fn_left[rows, pred_left] = 0
            fn_right = counts_right.copy()
            fn_right[rows, pred_right] = 0
            new_fn = base_fn[None, :] + fn_left + fn_right

            gains = parent_score - np.linalg.norm(self._qe(new_fp, new_fn), axis=1)
            errors = new_fn.sum(axis=1)
            # Highest gain first; among equal gains, fewest classification
            # errors (see the acceptance rule in ``_build``).
            best_row = int(np.lexsort((errors, -gains))[0])
            key = (gains[best_row], -errors[best_row])
            if best_key is None or key > best_key:
                best_key = key
                best = (float(gains[best_row]), float(errors[best_row]),
                        int(feature), float(thresholds[best_row]))

        return best

    def _route(self, node, X, indices, out, attr):
        if node.is_leaf():
            out[indices] = getattr(node, attr)
            return
        mask = X[indices, node.feature] <= node.threshold
        self._route(node.left, X, indices[mask], out, attr)
        self._route(node.right, X, indices[~mask], out, attr)

    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = check_array(X)
        out = np.zeros(X.shape[0], dtype=int)
        self._route(self.tree_, X, np.arange(X.shape[0]), out, "label")
        return self.classes_[out]

    def predict_proba(self, X):
        """Predict class probabilities as leaf training-class frequencies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class frequencies of the leaf each sample falls into.
        """
        X = check_array(X)
        out = np.zeros((X.shape[0], len(self.classes_)))
        self._route(self.tree_, X, np.arange(X.shape[0]), out, "proba")
        return out
