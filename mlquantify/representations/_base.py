

from abc import ABC, abstractmethod
import numpy as np

class BaseRepresentation(ABC):
    """Base class for quantification representations.
    
    Subclasses must implement :meth:`transform` and :meth:`_fit`.  The latter must set the ``class_representations_`` attribute to an array of shape (n_classes, n_representation_features) containing the representations for each class.

    Examples
    --------
    >>> from mlquantify.representations import BaseRepresentation
    >>> class DummyRepresentation(BaseRepresentation):
    ...     def _fit(self, X, y, sample_weight=None):
    ...         self.class_representations_ = np.array([[0], [1]])
    ...     def transform(self, X):
    ...         return X
    >>> X = [[0], [1], [0], [1]]
    >>> y = [0, 1, 0, 1]
    >>> rep = DummyRepresentation().fit(X, y)
    >>> rep.class_representations_
    array([[0],
           [1]])
    """

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


try:
    import torch
    import torch.nn as nn
    _torch_available = True
except ImportError:  # pragma: no cover - runtime-dependent
    _torch_available = False

    class _NNStub:
        class Module:
            pass

    nn = _NNStub()


class TorchRepresentation(nn.Module, ABC):
    """Base class for differentiable, permutation-invariant bag representations.

    Subclasses must implement :meth:`forward` and expose :attr:`output_size`.
    These representations are torch ``nn.Module`` instances — their parameters
    are learned jointly with the quantifier network during training.

    Unlike :class:`BaseRepresentation`, which is numpy-based and used by
    matching methods, ``TorchRepresentation`` is designed for neural
    quantifiers (HistNetQ, GMNet) that need end-to-end differentiability.

    The contract for ``forward``:

    - Input:  ``torch.Tensor`` of shape ``(n_examples, n_features)``
              representing one bag, or ``(n_bags, n_examples, n_features)`` for
              a batch of bags processed together (the neural quantifiers train
              on mini-batches of bags).
    - Output: ``torch.Tensor`` of shape ``(output_size,)`` for a single bag, or
              ``(n_bags, output_size)`` for a batch — a fixed-length
              permutation-invariant summary of each bag.

    Permutation invariance means ``forward(X) == forward(X[perm])`` for
    any permutation ``perm`` of the example axis (within each bag).
    """

    @abstractmethod
    def forward(self, X: "torch.Tensor") -> "torch.Tensor":
        """Map a bag (or batch of bags) to fixed-length representation vectors.

        Parameters
        ----------
        X : torch.Tensor of shape (n_examples, n_features) or \
            (n_bags, n_examples, n_features)
            Feature matrix for one bag, or a batch of bags.

        Returns
        -------
        repr : torch.Tensor of shape (output_size,) or (n_bags, output_size)
            Permutation-invariant bag representation(s).
        """

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Dimensionality of the output representation vector."""
