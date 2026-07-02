import queue
import threading

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - runtime-dependent
    torch = None


_PREFETCH_SENTINEL = object()


class BackgroundPrefetch:
    """Iterate ``iterable`` in a background thread, buffering up to ``size`` items.

    Used to overlap CPU-side bag sampling/stacking with GPU compute: while the
    main thread runs (and blocks on) the GPU, this worker thread produces the
    next mini-batches, so the GPU is not starved waiting on NumPy. The worker
    releases the GIL during the heavy NumPy calls and the main thread releases it
    while blocked on CUDA, so the two overlap.
    """

    def __init__(self, iterable, size=3):
        self._q = queue.Queue(maxsize=size)
        self._err = None
        self._thread = threading.Thread(target=self._worker, args=(iterable,), daemon=True)
        self._thread.start()

    def _worker(self, iterable):
        try:
            for item in iterable:
                self._q.put(item)
        except Exception as exc:  # surface producer errors to the consumer
            self._err = exc
        finally:
            self._q.put(_PREFETCH_SENTINEL)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is _PREFETCH_SENTINEL:
                if self._err is not None:
                    raise self._err
                return
            yield item


def rae_loss(p_pred, p_true, epsilon=None):
    """Smoothed Relative Absolute Error loss (Eq. 8, Pérez-Mon 2024).

    Parameters
    ----------
    p_pred : torch.Tensor of shape (1, n_classes) or (n_classes,)
    p_true : torch.Tensor of same shape
    epsilon : float or None
        Smoothing factor. If None, uses 1 / (2 * bag_size) with bag_size=500.

    Returns
    -------
    loss : torch.Tensor scalar
    """
    if epsilon is None:
        epsilon = 1.0 / (2 * 500)

    def smooth(p):
        n = p.shape[-1]
        return (p + epsilon) / (1 + n * epsilon)

    p_pred_s = smooth(p_pred)
    p_true_s = smooth(p_true)
    return torch.mean(torch.abs(p_pred_s - p_true_s) / p_true_s)


def bag_mixer(bags_X, bags_y, bags_prev, ratio=0.5, rng=None):
    """Bag Mixer augmentation (Section 3.1, Pérez-Mon 2024).

    Given a list of bags, randomly mix pairs to create new bags with
    interpolated prevalences.

    Parameters
    ----------
    bags_X : list of ndarray, each (bag_size, n_features)
    bags_y : list of ndarray, each (bag_size,) — individual labels
    bags_prev : list of ndarray, each (n_classes,)
    ratio : float, default=0.5
        Fraction of output bags that are mixed (rest are real bags).
    rng : numpy.random.Generator or None

    Returns
    -------
    out_X : list of ndarray
    out_prev : list of ndarray
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(bags_X)
    out_X, out_prev = [], []
    for i in range(n):
        if rng.random() < ratio:
            j = rng.integers(0, n)
            # Keep the mixed bag the same size as the original (so a batch of
            # bags can be stacked into one tensor): take half from bag i and the
            # remaining slots from bag j.
            size = len(bags_X[i])
            half_i = size // 2
            half_j = size - half_i
            idx_i = rng.choice(len(bags_X[i]), half_i, replace=False)
            idx_j = rng.choice(len(bags_X[j]), half_j, replace=True)
            mixed_X = np.concatenate([bags_X[i][idx_i], bags_X[j][idx_j]])
            mixed_prev = (
                half_i * bags_prev[i] + half_j * bags_prev[j]
            ) / size
            out_X.append(mixed_X)
            out_prev.append(mixed_prev)
        else:
            out_X.append(bags_X[i])
            out_prev.append(bags_prev[i])
    return out_X, out_prev


def sample_from_bag_pool(bags_X, bags_prev, n_bags, mix_ratio=0.5, rng=None):
    """Draw ``n_bags`` training bags from a pool of real prevalence-labelled bags.

    Used when training directly on bags labelled by prevalence (no per-example
    labels). Each output bag is, with probability ``mix_ratio``, a size-preserving
    Bag-Mixer mix of two random pool bags, and otherwise a real pool bag picked at
    random (Pérez-Mon 2024/2025; the authors' ``real_bags_proportion`` is
    ``1 - mix_ratio``).

    Parameters
    ----------
    bags_X : list of ndarray, each (bag_size, n_features)
        The pool of real bags (all the same size).
    bags_prev : list of ndarray, each (n_classes,)
        Prevalence label of each pool bag.
    n_bags : int
        Number of training bags to produce.
    mix_ratio : float, default=0.5
        Fraction of produced bags that are mixed (rest are real bags).
    rng : numpy.random.Generator or None

    Returns
    -------
    out_X : list of ndarray
    out_prev : list of ndarray
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(bags_X)
    out_X, out_prev = [], []
    for _ in range(n_bags):
        if n >= 2 and rng.random() < mix_ratio:
            i, j = rng.integers(0, n), rng.integers(0, n)
            size = len(bags_X[i])
            half_i = size // 2
            half_j = size - half_i
            idx_i = rng.choice(len(bags_X[i]), half_i, replace=True)
            idx_j = rng.choice(len(bags_X[j]), half_j, replace=True)
            out_X.append(np.concatenate([bags_X[i][idx_i], bags_X[j][idx_j]]))
            out_prev.append((half_i * bags_prev[i] + half_j * bags_prev[j]) / size)
        else:
            i = rng.integers(0, n)
            out_X.append(bags_X[i])
            out_prev.append(bags_prev[i])
    return out_X, out_prev


def cka_regularization(Z_list):
    """Centered Kernel Alignment diversity regularizer (Eq. 10, Pérez-Mon 2025).

    Encourages latent spaces to be diverse by penalizing similarity.
    Add to loss as: total_loss = task_loss + lambda * cka_regularization(Z_list)

    Parameters
    ----------
    Z_list : list of torch.Tensor, each (n_examples, d_l)
        Latent space projections for each of the L spaces.

    Returns
    -------
    cka_mean : torch.Tensor scalar (higher = more similar = worse)
    """
    L = len(Z_list)
    if L < 2:
        return torch.tensor(0.0)
    scores = []
    for i in range(L):
        for j in range(i + 1, L):
            Zi, Zj = Z_list[i], Z_list[j]
            # Frobenius-norm based CKA
            num = (Zi.T @ Zj).norm(p="fro") ** 2
            denom = (Zi.T @ Zi).norm(p="fro") * (Zj.T @ Zj).norm(p="fro")
            scores.append(num / (denom + 1e-8))
    return torch.stack(scores).mean()
