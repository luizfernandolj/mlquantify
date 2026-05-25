import numpy as np

from mlquantify.representations._base import BaseRepresentation


class PredictionRepresentation(BaseRepresentation):
    """Representation based on classifier predictions.

    Converts raw feature vectors (or pre-computed posterior probability
    arrays) into a fixed-length representation suitable for distribution
    matching.  Two built-in modes are provided:

    - ``'soft'``: treats each row of ``X`` as a soft posterior vector and
      returns the class-mean posterior (average soft predictions).
    - ``'hard'``: converts posterior or label arrays to a one-hot encoding
      and returns the empirical class-frequency vector (equivalent to
      Classify-and-Count applied to the representation).

    A custom transformation function can also be supplied via ``func``, and
    an optional downstream ``representation`` can further map the
    transformed outputs.

    Parameters
    ----------
    method : {'soft', 'hard'}, default='soft'
        Built-in transformation mode.  Ignored when ``func`` is provided.
    average : bool, default=True
        If ``True``, the representation is averaged over instances.
        If ``False``, the per-instance matrix is returned.
    func : callable or None, default=None
        Custom transformation ``func(X, representation) -> Z``.  When
        provided, ``method`` is ignored.
    representation : BaseRepresentation or None, default=None
        Optional nested representation applied to the transformed output
        ``Z`` before returning.

    Attributes
    ----------
    priors_ : ndarray of shape (n_classes,)
        Empirical class proportions observed during ``fit``.
    class_representations_ : ndarray
        Per-class mean representation vectors (or nested object array).
    classes_ : ndarray of shape (n_classes,)
        Unique class labels seen during ``fit``.

    Examples
    --------
    >>> from mlquantify.representations._prediction import SoftPredictionRepresentation
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> posteriors = rng.dirichlet([2, 2], size=100)
    >>> y = posteriors.argmax(axis=1)
    >>> rep = SoftPredictionRepresentation().fit(posteriors, y)
    >>> rep.transform(posteriors[:10]).shape
    (2,)
    """

    def __init__(self, method="soft", average=True, func=None, representation=None):
        self.method = method
        self.average = average
        self.func = func
        self.representation = representation

    def transform(self, X):
        """Transform instances into the prediction representation.

        Applies the configured transformation (soft/hard/custom) to ``X``
        and optionally passes the result through a nested representation.
        When ``average=True`` the output is a single mean vector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_classes) or (n_samples,)
            Posterior probabilities, raw scores, or hard labels.

        Returns
        -------
        representation : ndarray of shape (n_classes,) or \
            (n_samples, n_classes)
            Mean representation vector (when ``average=True``) or
            per-instance representation matrix.

        Examples
        --------
        >>> from mlquantify.representations._prediction import SoftPredictionRepresentation
        >>> import numpy as np
        >>> rng = np.random.default_rng(1)
        >>> posteriors = rng.dirichlet([1, 3], size=50)
        >>> y = posteriors.argmax(axis=1)
        >>> rep = SoftPredictionRepresentation().fit(posteriors, y)
        >>> rep.transform(posteriors[:5]).shape
        (2,)
        """
        Z = self._transformer()(X, self)

        if self.representation is not None:
            return self.representation.transform(Z)

        return self._aggregate(Z)

    def _aggregate(self, Z):
        if self.average:
            return np.mean(Z, axis=0)

        return Z

    def _fit(self, X, y, sample_weight=None):
        self.priors_ = np.asarray([
            np.mean(y == cls)
            for cls in self.classes_
        ])

        Z = self._transformer()(X, self)

        if self.representation is not None:
            self.representation.fit(
                Z,
                y,
                classes=self.classes_,
                sample_weight=sample_weight,
            )
            self.class_representations_ = self.representation.class_representations_
            return

        if self.average:
            self.class_representations_ = np.asarray([
                self._aggregate(Z[y == cls])
                for cls in self.classes_
            ], dtype=float)
        else:
            self.class_representations_ = np.empty(len(self.classes_), dtype=object)

            for class_idx, cls in enumerate(self.classes_):
                self.class_representations_[class_idx] = self._aggregate(Z[y == cls])

    def _transformer(self):
        if self.func is not None:
            return self.func

        transformers = {
            "soft": self._soft_predictions,
            "hard": self._hard_predictions,
        }

        try:
            return transformers[self.method]
        except KeyError as exc:
            raise ValueError(
                f"Unknown prediction representation method {self.method!r}. "
                f"Expected one of {sorted(transformers)} or pass func=..."
            ) from exc

    @staticmethod
    def _soft_predictions(X, representation):
        return np.asarray(X, dtype=float)

    @staticmethod
    def _hard_predictions(X, representation):
        X = np.asarray(X)

        if X.ndim == 2:
            label_indices = np.argmax(X, axis=1)
            labels = representation.classes_[label_indices]
        else:
            labels = X

        onehot = np.zeros((len(labels), len(representation.classes_)), dtype=float)

        for class_idx, cls in enumerate(representation.classes_):
            onehot[:, class_idx] = labels == cls

        return onehot

    def class_likelihoods(self, X):
        """Retrieve per-class likelihoods from the nested representation.

        Delegates to the ``class_likelihoods`` method of the configured
        nested ``representation``.  Only available when a nested
        representation that exposes ``class_likelihoods`` (e.g.
        :class:`~mlquantify.representations.KDERepresentation`) was
        supplied at construction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test instances (after the primary transformation).

        Returns
        -------
        likelihoods : ndarray of shape (n_classes, n_samples)
            Per-class likelihoods for every test instance.

        Raises
        ------
        AttributeError
            If no nested representation is configured, or if the nested
            representation does not implement ``class_likelihoods``.

        Examples
        --------
        >>> from mlquantify.representations._prediction import SoftPredictionRepresentation
        >>> from mlquantify.representations._density import KDERepresentation
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> posteriors = rng.dirichlet([2, 2], size=100)
        >>> y = posteriors.argmax(axis=1)
        >>> kde = KDERepresentation(bandwidth=0.1)
        >>> rep = SoftPredictionRepresentation(representation=kde).fit(posteriors, y)
        >>> rep.class_likelihoods(posteriors[:5]).shape
        (2, 5)
        """
        if self.representation is None or not hasattr(self.representation, "class_likelihoods"):
            raise AttributeError(
                f"{type(self).__name__} does not expose class likelihoods without "
                "a nested likelihood representation."
            )

        return self.representation.class_likelihoods(X)


class HardPredictionRepresentation(PredictionRepresentation):
    """Hard-prediction representation convenience class.

    Shortcut for ``PredictionRepresentation(method='hard')``.  Converts
    posterior arrays or label vectors to one-hot encodings and returns
    the mean, which is equivalent to the empirical class frequency.

    Parameters
    ----------
    average : bool, default=True
        If ``True``, return the mean one-hot vector (class frequencies).

    Examples
    --------
    >>> from mlquantify.representations._prediction import HardPredictionRepresentation
    >>> import numpy as np
    >>> posteriors = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    >>> y = np.array([0, 1, 0])
    >>> rep = HardPredictionRepresentation().fit(posteriors, y)
    >>> rep.transform(posteriors).shape
    (2,)
    """

    def __init__(self, average=True):
        super().__init__(method="hard", average=average)


class SoftPredictionRepresentation(PredictionRepresentation):
    """Soft-prediction representation convenience class.

    Shortcut for ``PredictionRepresentation(method='soft')``.  Treats
    each row of the input as a posterior probability vector and returns
    the class-mean posterior, which is the representation used by
    Probabilistic Classify and Count (PCC).

    Parameters
    ----------
    average : bool, default=True
        If ``True``, return the mean posterior vector.

    Examples
    --------
    >>> from mlquantify.representations._prediction import SoftPredictionRepresentation
    >>> import numpy as np
    >>> posteriors = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    >>> y = np.array([0, 1, 0])
    >>> rep = SoftPredictionRepresentation().fit(posteriors, y)
    >>> rep.transform(posteriors)
    array([0.5, 0.5])
    """

    def __init__(self, average=True):
        super().__init__(method="soft", average=average)
