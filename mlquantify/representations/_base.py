

from abc import ABC, abstractmethod
import numpy as np

class BaseRepresentation(ABC):
    """Base class for quantification representations."""

    def fit(self, X, y, classes=None, sample_weight=None):
        """Fit the representation to labelled training data.

        Validates shapes, stores the class labels, delegates internal
        fitting to :meth:`_fit`, and verifies that the subclass set
        ``class_representations_`` during that call.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or \
            (n_samples,)
            Feature matrix or pre-computed score array for the training
            instances.
        y : array-like of shape (n_samples,)
            Class labels for each training instance.
        classes : array-like of shape (n_classes,) or None, default=None
            Explicit list of class labels.  If ``None``, the unique
            values in ``y`` are used.
        sample_weight : array-like of shape (n_samples,) or None, \
            default=None
            Per-sample weights forwarded to :meth:`_fit`.

        Returns
        -------
        self : BaseRepresentation
            The fitted representation object.

        Raises
        ------
        ValueError
            If ``X`` and ``y`` have inconsistent lengths or ``X`` is
            zero-dimensional.
        AttributeError
            If the subclass did not define ``class_representations_``
            inside :meth:`_fit`.

        Examples
        --------
        >>> from mlquantify.representations import HistogramRepresentation
        >>> import numpy as np
        >>> X = np.random.default_rng(0).uniform(0, 1, (100, 1))
        >>> y = (X[:, 0] > 0.5).astype(int)
        >>> rep = HistogramRepresentation(bins=(5,)).fit(X, y)
        >>> rep.class_representations_.shape
        (2, 5)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 0:
            raise ValueError("X must be array-like.")

        if y.ndim != 1:
            y = y.ravel()

        if len(X) != len(y):
            raise ValueError(
                f"X and y have inconsistent lengths: {len(X)} and {len(y)}."
            )

        self.classes_ = np.asarray(classes) if classes is not None else np.unique(y)
        self._fit(X, y, sample_weight=sample_weight)

        if not hasattr(self, "class_representations_"):
            raise AttributeError(
                f"{type(self).__name__} must define class_representations_ in _fit()."
            )

        return self

    @abstractmethod
    def transform(self, X):
        """Transform data into the representation space."""

    @abstractmethod
    def _fit(self, X, y, sample_weight=None):
        """Fit representation internals."""
