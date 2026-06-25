"""Synthetic sample generators for quantification experiments."""

from collections import namedtuple

import numpy as np

#: Per-bag linear decision boundary returned by ``make_quantification`` when
#: ``return_boundary=True``. ``coef`` and ``intercept`` are stacked over bags:
#: for binary problems ``coef`` is ``(n_bags, n_features)`` and ``intercept`` is
#: ``(n_bags,)``; for multiclass, ``(n_bags, n_classes, n_features)`` and
#: ``(n_bags, n_classes)``. Each bag's labels satisfy ``coef[i] . x + intercept[i]``
#: (sign for binary, argmax for multiclass). Covariate/prior bags share one fixed
#: boundary across rows; concept bags each carry their own rotated boundary.
DecisionBoundary = namedtuple("DecisionBoundary", ["coef", "intercept"])

from mlquantify.utils._random import check_random_state
from mlquantify.utils._sampling import (
    get_indexes_with_prevalence,
    simplex_dirichlet_sampling,
    simplex_grid_sampling,
    simplex_uniform_kraemer,
)


def _auto_n_informative(n_classes, n_clusters_per_class, n_features, n_redundant):
    """Smallest ``n_informative`` that lets make_classification place the clusters."""
    need = int(np.ceil(np.log2(max(2, n_classes * n_clusters_per_class)))) + 1
    n_informative = max(2, need)
    return min(n_informative, n_features - n_redundant)


def _resolve_sizes(batch_size, n_bags, rng):
    """Turn ``batch_size`` (int, ``(low, high)`` range, or sequence) into a list."""
    if isinstance(batch_size, (int, np.integer)):
        return [int(batch_size)] * n_bags
    if isinstance(batch_size, tuple) and len(batch_size) == 2:
        low, high = batch_size
        return [int(rng.integers(low, high + 1)) for _ in range(n_bags)]
    sizes = [int(s) for s in batch_size]
    if len(sizes) != n_bags:
        raise ValueError(
            f"batch_size has length {len(sizes)} but {n_bags} bags were requested."
        )
    return sizes


def _make_prevalences(
    prevalence, n_classes, n_batches, *, target_prevalence, concentration,
    min_prev, max_prev, n_prevalences, repeats, rng,
):
    """Return an array of target prevalence vectors, or ``None`` for ``'natural'``.

    Encapsulates the four prior-shift sampling strategies. The ``'natural'``
    strategy returns ``None`` to signal that bags are drawn i.i.d. from the
    population instead of toward a target prevalence.
    """
    # An explicit array of prevalence vectors.
    if not isinstance(prevalence, str):
        prevs = np.atleast_2d(np.asarray(prevalence, dtype=float))
        if prevs.shape[1] != n_classes:
            raise ValueError(
                f"Explicit prevalence vectors must have {n_classes} columns, "
                f"got shape {prevs.shape}."
            )
        if not np.allclose(prevs.sum(axis=1), 1.0):
            raise ValueError("Each explicit prevalence vector must sum to 1.")
        return prevs

    strategy = prevalence.lower()
    if strategy == "natural":
        return None
    if strategy == "uniform":
        return simplex_uniform_kraemer(
            n_classes, n_batches, 1, min_prev, max_prev, random_state=rng,
        )
    if strategy == "grid":
        npv = n_prevalences if n_prevalences is not None else (11 if n_classes == 2 else 5)
        return simplex_grid_sampling(n_classes, npv, repeats, min_prev, max_prev)
    if strategy == "dirichlet":
        target = (
            np.full(n_classes, 1.0 / n_classes) if target_prevalence is None
            else np.asarray(target_prevalence, dtype=float)
        )
        if target.shape != (n_classes,):
            raise ValueError(
                f"target_prevalence must have length {n_classes}."
            )
        kappa = float(concentration) if concentration is not None else float(n_classes)
        alpha = kappa * target
        return simplex_dirichlet_sampling(
            n_classes, n_batches, 1, alpha=alpha,
            min_val=min_prev, max_val=max_prev, random_state=rng,
        )
    raise ValueError(
        f"Unknown prevalence strategy {prevalence!r}. Expected one of "
        "'uniform', 'grid', 'natural', 'dirichlet', or an array of vectors."
    )


def _normalize_shift_type(shift_type):
    """Coerce ``shift_type`` (str or iterable) into a validated list of shifts."""
    shifts = [shift_type] if isinstance(shift_type, str) else list(shift_type)
    valid = {"prior", "covariate", "concept"}
    if not shifts:
        raise ValueError("shift_type must name at least one shift.")
    for s in shifts:
        if s not in valid:
            raise ValueError(
                f"shift_type entries must be in {sorted(valid)}; got {s!r}."
            )
    return shifts


def _boundary_labels(reference, X, rotate, strength, rng):
    """Label ``X`` by the reference boundary, optionally rotated by ``strength``.

    With ``rotate=False`` this is the fixed reference rule (covariate shift uses
    it after translating the features). With ``rotate=True`` the boundary is
    moved per call (concept shift), so the same point can change label.

    Returns ``(labels, W, b)`` where ``(W, b)`` is the linear boundary actually
    used (``W`` of shape ``(n_outputs, n_features)``), so callers can report it.
    """
    W = np.atleast_2d(reference.coef_).astype(float)
    b = np.atleast_1d(reference.intercept_).astype(float)

    if rotate:
        if W.shape[0] == 1:
            # Binary: rotate the weight vector inside a random plane.
            w = W[0]
            theta = strength * rng.uniform(-1.0, 1.0)
            ortho = rng.standard_normal(w.shape)
            ortho = ortho - (ortho @ w) / (w @ w) * w
            norm = np.linalg.norm(ortho)
            if norm > 1e-12:
                w = np.cos(theta) * w + np.sin(theta) * np.linalg.norm(w) * ortho / norm
            W = w[None, :]
        else:
            # Multiclass: perturb the whole coefficient matrix coherently.
            noise = rng.standard_normal(W.shape)
            step = strength * np.linalg.norm(W) / (np.linalg.norm(noise) + 1e-12)
            W = W + step * noise

    scores = X @ W.T + b
    labels_idx = (scores[:, 0] > 0).astype(int) if W.shape[0] == 1 else scores.argmax(axis=1)
    return reference.classes_[labels_idx], W, b


def make_quantification(
    n_batches=10,
    batch_size=500,
    *,
    n_samples=10000,
    n_classes=2,
    n_features=20,
    n_informative=None,
    n_redundant=2,
    n_clusters_per_class=1,
    class_sep=1.0,
    flip_y=0.01,
    weights=None,
    shift_type="prior",
    prevalence="uniform",
    target_prevalence=None,
    concentration=None,
    min_prev=0.0,
    max_prev=1.0,
    n_prevalences=None,
    repeats=1,
    covariate_scale=None,
    concept_strength=None,
    return_train=False,
    train_size=None,
    train_prevalence=None,
    return_prevalences=True,
    return_boundary=False,
    stack=False,
    pack="lists",
    as_frame=False,
    shuffle=True,
    random_state=None,
):
    r"""Generate synthetic quantification bags under prior-probability shift.

    The quantification analogue of :func:`sklearn.datasets.make_classification`:
    it builds one labelled population, then draws ``n_batches`` *bags* from it,
    where each bag's class prevalence is sampled according to a shift strategy.
    Because the quantification target is a bag's class distribution (not its
    per-instance labels), the true prevalence of every bag is returned alongside
    the data.

    Three kinds of dataset shift are supported through ``shift_type``:

    - **prior** — :math:`P(y)` changes (bags resampled to a target prevalence);
      the clusters keep their position.
    - **covariate** — the *position of the features* changes while the decision
      boundary stays fixed: each bag's feature cloud is translated, then labelled
      by the same fixed boundary, so a class appears in new regions of space.
    - **concept** — the *decision boundary moves*: points stay where they are and
      are relabelled by a per-bag rotation of a reference boundary.

    In every case the returned ``prevalences`` are the achieved class proportions
    of each bag.

    Parameters
    ----------
    n_batches : int, default=10
        Number of bags to draw. Ignored when ``prevalence='grid'`` (the grid
        density then sets the count).
    batch_size : int, tuple(low, high), or sequence of int, default=500
        Size of each bag. An int gives equal sizes; a ``(low, high)`` tuple
        draws a random size per bag; a sequence sets each size explicitly.
    n_samples : int, default=10000
        Size of the underlying labelled population the bags are drawn from.
    n_classes : int, default=2
        Number of classes.
    n_features, n_informative, n_redundant, n_clusters_per_class : int
        Passed to :func:`~sklearn.datasets.make_classification`. When
        ``n_informative`` is ``None`` a value large enough to place the class
        clusters is chosen automatically.
    class_sep : float, default=1.0
        Class separability. Lower values make quantification harder (and the
        adjustment of ACC/EMQ/DyS more valuable).
    flip_y : float, default=0.01
        Fraction of labels randomly flipped in the population (label noise).
    weights : array-like of shape (n_classes,), default=None
        Class balance of the population (its prior :math:`P(y)`).
    shift_type : str or list of str, default='prior'
        One of ``'prior'``, ``'covariate'``, ``'concept'`` (see the summary
        above), or a **list of them to stack** for a more diverse dataset, e.g.
        ``['prior', 'covariate']``. When stacked they compose per bag: covariate
        translates the features, concept rotates the labelling boundary, and
        prior resamples to a target prevalence. ``prevalence`` applies to
        ``'prior'``; ``covariate_scale`` / ``concept_strength`` tune the others.
    prevalence : {'uniform', 'grid', 'natural', 'dirichlet'} or array-like, default='uniform'
        For ``shift_type='prior'`` — how each bag's target prevalence is sampled:

        - ``'uniform'`` — uniformly over the probability simplex (full range of
          shifts).
        - ``'grid'`` — a regular grid over the simplex (the Artificial
          Prevalence Protocol); count set by ``n_prevalences`` / ``repeats``.
        - ``'natural'`` — bags drawn i.i.d. from the population, so prevalence
          fluctuates around ``weights`` with sampling noise only.
        - ``'dirichlet'`` — from a Dirichlet centred on ``target_prevalence``
          with spread controlled by ``concentration``.
        - an array of shape ``(n_batches, n_classes)`` of explicit vectors.
    target_prevalence : array-like of shape (n_classes,), default=None
        Mean prevalence for ``prevalence='dirichlet'`` (defaults to balanced).
    concentration : float, default=None
        Dirichlet total concentration :math:`\kappa`. Larger values keep bags
        tightly around ``target_prevalence`` (low variability); smaller values
        spread them toward extreme shifts. Defaults to ``n_classes`` (which
        reproduces the uniform simplex when ``target_prevalence`` is balanced).
    min_prev, max_prev : float, default=0.0, 1.0
        Per-class clipping bounds on the sampled prevalences.
    n_prevalences, repeats : int, default=None, 1
        Grid density and repetitions for ``prevalence='grid'``.
    covariate_scale : float, default=None
        For ``shift_type='covariate'`` — magnitude of the per-bag feature
        translation (in feature-std units). ``0`` leaves the cloud in place;
        larger values move ``P(x)`` further. Defaults to 1.5.
    concept_strength : float, default=None
        For ``shift_type='concept'`` — how far the reference decision boundary is
        rotated per bag (radians-scale). ``0`` keeps the base boundary; larger
        values move it more. Defaults to 0.5.
    return_train : bool, default=False
        If ``True``, also return a dedicated training sample drawn from a
        disjoint half of the population.
    train_size : int, default=None
        Size of the returned training sample (defaults to the whole training
        half).
    train_prevalence : array-like of shape (n_classes,), default=None
        Prevalence of the training sample (defaults to the natural population
        prior).
    return_prevalences : bool, default=True
        If ``True``, also return the ``(n_bags, n_classes)`` array of each
        bag's true prevalence.
    return_boundary : bool, default=False
        If ``True``, also return a ``DecisionBoundary`` namedtuple capturing
        the linear boundary used for **each bag** (``coef`` and ``intercept``
        stacked over bags). Covariate and prior bags share one fixed boundary;
        concept bags each carry their *own rotated* boundary — so the object
        records exactly how the rule moves, with no need to re-fit a classifier.
        For 2-D data, draw bag ``i`` from ``coef[i]`` / ``intercept[i]``.
    stack : bool, default=False
        If ``True``, stack the bags into ``(n_bags, batch_size, n_features)``
        and ``(n_bags, batch_size)`` arrays. Requires equal bag sizes.
    pack : {'lists', 'flat'}, default='lists'
        ``'lists'`` returns ``Xs`` and ``ys`` as lists of per-bag arrays.
        ``'flat'`` spreads them as ``(X1, ..., Xn, y1, ..., yn)``
        (incompatible with ``return_train``).
    as_frame : bool, default=False
        Return each bag's ``X`` as a pandas ``DataFrame`` and ``y`` as a
        ``Series``.
    shuffle : bool, default=True
        Shuffle instances within each bag.
    random_state : int, Generator or None, default=None
        Controls the population and the sampling.

    Returns
    -------
    The return is a tuple assembled from the following, in order: the optional
    training sample ``X_train, y_train`` (only if ``return_train``), the bags
    ``Xs, ys`` (lists, or stacked arrays, or spread out when ``pack='flat'``),
    the ``prevalences`` array (only if ``return_prevalences``), and the fitted
    decision ``boundary`` (only if ``return_boundary``). With the defaults this
    is ``(Xs, ys, prevalences)``.

    See Also
    --------
    mlquantify.model_selection.apply_protocol : Run a protocol over real data.

    Examples
    --------
    >>> from mlquantify.datasets import make_quantification
    >>> Xs, ys, prevs = make_quantification(n_batches=3, random_state=0)
    >>> len(Xs), len(ys), prevs.shape
    (3, 3, (3, 2))
    >>> # Bags concentrated near a 70/30 split:
    >>> Xs, ys, prevs = make_quantification(   # doctest: +SKIP
    ...     n_batches=20, prevalence="dirichlet",
    ...     target_prevalence=[0.7, 0.3], concentration=150, random_state=0)
    """
    shifts = _normalize_shift_type(shift_type)
    use_prior = "prior" in shifts
    use_cov = "covariate" in shifts
    use_concept = "concept" in shifts
    if pack not in ("lists", "flat"):
        raise ValueError("pack must be 'lists' or 'flat'.")
    if pack == "flat" and return_train:
        raise ValueError("pack='flat' is incompatible with return_train=True.")

    rng = check_random_state(random_state)

    if n_informative is None:
        n_informative = _auto_n_informative(
            n_classes, n_clusters_per_class, n_features, n_redundant
        )

    from sklearn.datasets import make_classification

    X_pool, y_pool = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class,
        weights=weights,
        class_sep=class_sep,
        flip_y=flip_y,
        shuffle=True,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )

    # Optionally carve out a disjoint training half.
    X_train = y_train = None
    if return_train:
        perm = rng.permutation(n_samples)
        cut = n_samples // 2
        tr_idx, te_idx = perm[:cut], perm[cut:]
        Xtr_pool, ytr_pool = X_pool[tr_idx], y_pool[tr_idx]
        src_X, src_y = X_pool[te_idx], y_pool[te_idx]
        if train_prevalence is None:
            X_train, y_train = Xtr_pool, ytr_pool
        else:
            idx = get_indexes_with_prevalence(
                ytr_pool, list(train_prevalence),
                train_size or len(ytr_pool), random_state=rng,
            )
            X_train, y_train = Xtr_pool[idx], ytr_pool[idx]
    else:
        src_X, src_y = X_pool, y_pool

    classes = np.unique(y_pool)

    # Prior shift sets the per-bag target prevalence (and may set the bag count).
    if use_prior:
        prevs_target = _make_prevalences(
            prevalence, n_classes, n_batches,
            target_prevalence=target_prevalence, concentration=concentration,
            min_prev=min_prev, max_prev=max_prev,
            n_prevalences=n_prevalences, repeats=repeats, rng=rng,
        )
        natural = prevs_target is None
        n_bags = n_batches if natural else len(prevs_target)
    else:
        prevs_target, natural, n_bags = None, False, n_batches
    sizes = _resolve_sizes(batch_size, n_bags, rng)

    # Resolve the per-shift strength knobs and fit the reference boundary once.
    cov_scale = 1.5 if covariate_scale is None else float(covariate_scale)
    concept_str = 0.5 if concept_strength is None else float(concept_strength)
    reference = None
    if use_cov or use_concept or return_boundary:
        from sklearn.linear_model import LogisticRegression

        # The population's linear decision boundary. Covariate shift moves the
        # features across it; concept shift rotates it; it is returned to the
        # caller when ``return_boundary=True`` so the labelling rule is known.
        reference = LogisticRegression(max_iter=1000).fit(X_pool, y_pool)

    # Each bag composes the requested shifts: covariate translates the features,
    # concept rotates the labelling boundary, and prior resamples to a target
    # prevalence. Labels come from the (rotated) boundary whenever covariate or
    # concept is active, otherwise from the population's own labels.
    Xs, ys, true_prev = [], [], []
    bnd_coef, bnd_int = [], []   # per-bag boundary, collected when return_boundary
    for i in range(n_bags):
        bsize = sizes[i]

        cand_X = src_X
        if use_cov:
            cand_X = np.asarray(src_X, dtype=float) + cov_scale * rng.standard_normal(
                src_X.shape[1]
            )
        if use_cov or use_concept:
            cand_y, W_i, b_i = _boundary_labels(
                reference, np.asarray(cand_X, dtype=float),
                use_concept, concept_str, rng,
            )
        else:
            cand_y = src_y
            if reference is not None:   # prior-only with return_boundary
                W_i = np.atleast_2d(reference.coef_).astype(float)
                b_i = np.atleast_1d(reference.intercept_).astype(float)

        if use_prior and not natural and len(np.unique(cand_y)) == n_classes:
            idx = np.asarray(get_indexes_with_prevalence(
                cand_y, list(prevs_target[i]), bsize, random_state=rng,
            ))
        else:
            idx = rng.choice(len(cand_y), size=bsize, replace=True)
        if shuffle:
            rng.shuffle(idx)

        Xb, yb = np.asarray(cand_X)[idx], np.asarray(cand_y)[idx]
        Xs.append(Xb)
        ys.append(yb)
        true_prev.append([float(np.mean(yb == c)) for c in classes])
        if return_boundary:
            bnd_coef.append(W_i)
            bnd_int.append(b_i)

    prevalences = np.asarray(true_prev)

    boundary = None
    if return_boundary:
        coef = np.stack(bnd_coef)        # (n_bags, n_outputs, n_features)
        intercept = np.stack(bnd_int)    # (n_bags, n_outputs)
        if coef.shape[1] == 1:           # binary: drop the singleton output axis
            coef = coef[:, 0, :]
            intercept = intercept[:, 0]
        boundary = DecisionBoundary(coef=coef, intercept=intercept)

    if as_frame:
        import pandas as pd

        cols = [f"feature_{j}" for j in range(n_features)]
        Xs = [pd.DataFrame(Xb, columns=cols) for Xb in Xs]
        ys = [pd.Series(yb, name="target") for yb in ys]
        if X_train is not None:
            X_train = pd.DataFrame(X_train, columns=cols)
            y_train = pd.Series(y_train, name="target")

    if stack:
        if len(set(sizes)) != 1:
            raise ValueError("stack=True requires all bags to have equal size.")
        if as_frame:
            raise ValueError("stack=True is incompatible with as_frame=True.")
        Xs = np.stack(Xs)
        ys = np.stack(ys)

    # Assemble the return tuple.
    if pack == "flat":
        out = (*Xs, *ys)
        if return_prevalences:
            out = (*out, prevalences)
        if return_boundary:
            out = (*out, boundary)
        return out

    out = []
    if return_train:
        out += [X_train, y_train]
    out += [Xs, ys]
    if return_prevalences:
        out.append(prevalences)
    if return_boundary:
        out.append(boundary)
    return tuple(out)
