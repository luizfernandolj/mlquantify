"""Training symmetric neural quantifiers directly on prevalence-labelled bags.

The default :class:`~mlquantify.neural.HistNetQ` / :class:`~mlquantify.neural.GMNet`
fit path takes individually labelled data ``(X, y)`` and composes artificial bags
from it (APP). :class:`PrevalenceBagMixin` swaps that for the *symmetric* training
regime of Pérez-Mon et al. (2024, 2025): the training data is a **collection of
bags, each labelled only by its class prevalence** (no per-example labels), as in
the LeQua competitions. New bags are synthesised by mixing the real ones
(Bag Mixer), and the quantification loss is optimised directly.
"""

import numpy as np

from mlquantify.neural._utils import sample_from_bag_pool
from mlquantify.neural._classes import HistNetQ, GMNet


class PrevalenceBagMixin:
    """Mixin: fit a symmetric neural quantifier on bags labelled by prevalence.

    Mix this into :class:`~mlquantify.neural.HistNetQ` or
    :class:`~mlquantify.neural.GMNet` to replace their ``fit(X, y)`` with
    ``fit(Xs, ps)``, where ``Xs`` is a collection of bags and ``ps`` their
    prevalence labels. The ready-made
    :class:`~mlquantify.neural.HistNetQBags` and
    :class:`~mlquantify.neural.GMNetBags` are exactly these compositions.

    Everything after data ingestion — the network, the differentiable bag
    representation, the mini-batched training loop, early stopping and
    :meth:`predict` — is inherited unchanged from the host quantifier.

    Notes
    -----
    All bags must share the same size so they can be stacked into mini-batches.
    Pass ``bag_size`` to subsample every bag to a common length when they differ
    (``None`` keeps the bags as given and requires them to be equal-sized).
    """

    def fit(self, Xs, ps, classes=None):
        """Fit on a collection of prevalence-labelled bags.

        Parameters
        ----------
        Xs : list of array-like, or ndarray of shape (n_bags, bag_size, n_features)
            The training bags. Each bag is a ``(n_examples, n_features)`` matrix;
            a single 3-D array is also accepted.
        ps : array-like of shape (n_bags, n_classes)
            Class prevalence of each bag. Rows are renormalised to sum to one.
        classes : array-like of shape (n_classes,) or None, default=None
            Class labels in column order. Defaults to ``range(n_classes)``.

        Returns
        -------
        self
        """
        bags, prevs = self._validate_bags(Xs, ps)
        n_classes = prevs.shape[1]
        self.classes_ = (
            np.asarray(classes) if classes is not None else np.arange(n_classes)
        )
        rng = np.random.default_rng(self.random_state)

        # Standardise features at the data level over all examples in all bags
        # (a fixed mean/std), mirroring the (X, y) path. Per-bag normalisation
        # would erase the prevalence signal carried by each bag's composition.
        stacked = np.concatenate(bags, axis=0)
        self._x_mean = stacked.mean(axis=0, keepdims=True)
        self._x_std = stacked.std(axis=0, keepdims=True) + 1e-8
        bags = [(b - self._x_mean) / self._x_std for b in bags]

        # Split the *bags* (not examples) into train / validation pools.
        n = len(bags)
        perm = rng.permutation(n)
        n_val = max(1, int(round(0.2 * n)))
        val_idx, tr_idx = perm[:n_val], perm[n_val:]
        self._pool_X = [bags[i] for i in tr_idx]
        self._pool_prev = [prevs[i] for i in tr_idx]
        self._val_bags = [(bags[i], prevs[i]) for i in val_idx]

        n_features = bags[0].shape[1]
        return self._fit_loop(n_features, n_classes, rng)

    def _sample_bag_chunk(self, rng, k):
        """Draw ``k`` training bags from the real-bag pool + Bag Mixer."""
        return sample_from_bag_pool(
            self._pool_X, self._pool_prev, k,
            mix_ratio=self.bag_mixer_ratio, rng=rng,
        )

    def _validate_bags(self, Xs, ps):
        """Coerce ``Xs`` to a list of equal-sized 2-D float arrays and check ``ps``."""
        if isinstance(Xs, np.ndarray) and Xs.ndim == 3:
            bags = [np.asarray(b, dtype=float) for b in Xs]
        else:
            bags = [np.asarray(b, dtype=float) for b in Xs]
        if any(b.ndim != 2 for b in bags):
            raise ValueError("Each bag in Xs must be 2-D (n_examples, n_features).")

        prevs = np.asarray(ps, dtype=float)
        if prevs.ndim != 2:
            raise ValueError("ps must be 2-D of shape (n_bags, n_classes).")
        if len(bags) != len(prevs):
            raise ValueError(
                f"Xs and ps have inconsistent lengths: {len(bags)} and {len(prevs)}."
            )

        n_features = bags[0].shape[1]
        if any(b.shape[1] != n_features for b in bags):
            raise ValueError("All bags must have the same number of features.")

        # Optionally subsample every bag to a common size so they can be batched.
        bag_size = getattr(self, "bag_size", None)
        sizes = {b.shape[0] for b in bags}
        if bag_size is not None and len(sizes) > 0:
            bags = [self._resize_bag(b, bag_size, np.random.default_rng(self.random_state))
                    for b in bags]
        elif len(sizes) > 1:
            raise ValueError(
                "All bags must have the same size to be stacked into mini-batches; "
                "got sizes %s. Pass bag_size= to subsample them to a common length."
                % sorted(sizes)
            )

        rowsum = prevs.sum(axis=1, keepdims=True)
        rowsum[rowsum == 0] = 1.0
        prevs = prevs / rowsum
        return bags, prevs

    @staticmethod
    def _resize_bag(bag, bag_size, rng):
        n = bag.shape[0]
        if n == bag_size:
            return bag
        idx = rng.choice(n, size=bag_size, replace=n < bag_size)
        return bag[idx]


class HistNetQBags(PrevalenceBagMixin, HistNetQ):
    """:class:`~mlquantify.neural.HistNetQ` trained on prevalence-labelled bags.

    Same network and hyper-parameters as :class:`~mlquantify.neural.HistNetQ`,
    but ``fit(Xs, ps)`` consumes a collection of bags with prevalence labels
    instead of ``fit(X, y)``. See :class:`PrevalenceBagMixin`.
    """


class GMNetBags(PrevalenceBagMixin, GMNet):
    """:class:`~mlquantify.neural.GMNet` trained on prevalence-labelled bags.

    Same network and hyper-parameters as :class:`~mlquantify.neural.GMNet`, but
    ``fit(Xs, ps)`` consumes a collection of bags with prevalence labels instead
    of ``fit(X, y)``. See :class:`PrevalenceBagMixin`.
    """
