import numpy as np

from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from mlquantify.base import BaseQuantifier, ProtocolMixin
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils._sampling import (
    get_indexes_with_prevalence,
    simplex_grid_sampling,
    simplex_uniform_kraemer,
    simplex_uniform_sampling,
    simplex_dirichlet_sampling,
)
from mlquantify.utils._random import check_random_state
from mlquantify.utils._validation import validate_data
from mlquantify.utils.prevalence import get_prev_from_labels
from abc import ABC, abstractmethod
from logging import warning
import numpy as np

    
class BaseProtocol(ProtocolMixin, BaseQuantifier):
    r"""Abstract base class for evaluation protocols.

    Provides the :meth:`split` interface that yields sample indices for
    evaluating a quantifier across varying prevalence conditions. Subclasses
    implement :meth:`_iter_indices` to define the specific sampling strategy.

    Parameters
    ----------
    batch_size : int or list of int
        Size(s) of the evaluation batches.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    n_combinations : int
        Total number of batches this protocol will generate.

    Examples
    --------
    >>> from mlquantify.model_selection._protocol import BaseProtocol
    >>> import numpy as np
    >>> class MyProtocol(BaseProtocol):
    ...     def _iter_indices(self, X, y):
    ...         rng = np.random.default_rng(self.random_state)
    ...         for bs in self.batch_size:
    ...             yield rng.choice(len(X), bs, replace=True)
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = MyProtocol(batch_size=50, random_state=0)
    >>> idx = next(proto.split(X, y))
    >>> len(idx)
    50
    """
    
    _parameter_constraints = {
        "batch_size": [Interval(left=1, right=None, discrete=True)],
        "random_state": [Interval(left=0, right=None, discrete=True)]
    }

    def __init__(self, batch_size, random_state=None, **kwargs):
        if isinstance(batch_size, int):
            self.n_combinations = 1
        else:
            self.n_combinations = len(batch_size)

        self.batch_size = [batch_size] if isinstance(batch_size, int) else batch_size
        self.random_state = random_state

        for name, value in kwargs.items():
            setattr(self, name, value)
            if isinstance(value, list):
                self.n_combinations *= len(value)
            elif isinstance(value, (int, float)):
                self.n_combinations *= value
            else:
                raise ValueError(f"Invalid argument {name}={value}: must be int/float or list of int/float.")
                 

    def split(self, X: np.ndarray, y: np.ndarray):
        r"""
        Split the data into samples for evaluation.

        Parameters
        ----------
        X : np.ndarray
            The input features.
        y : np.ndarray
            The target labels.

        Yields
        ------
        Generator[np.ndarray, np.ndarray]
            A generator that yields the indices for each split.
        """
        X, y = validate_data(self, X, y)
        for idx in self._iter_indices(X, y):
            if len(idx) > len(X):
                warning(f"Batch size {len(idx)} exceeds dataset size {len(X)}. Replacement sampling will be used.")
            yield idx


    @abstractmethod
    def _iter_indices(self, X, y):
        """Abstract method to be implemented by subclasses to yield indices for each batch."""
        pass
    
    def get_n_combinations(self):
        """
        Get the number of combinations for the current protocol.
        """
        return self.n_combinations



# ===========================================
# Protocol Implementations
# ===========================================


class APP(BaseProtocol):
    r"""Artificial Prevalence Protocol (APP).

    Generates evaluation batches with artificially imposed prevalences drawn
    from the probability simplex within ``[min_prev, max_prev]``, covering a
    range of prevalence levels for comprehensive evaluation. The way the
    prevalence points are produced is controlled by the ``strategy`` parameter.

    Parameters
    ----------
    batch_size : int or list of int
        Size(s) of the evaluation batches.
    n_prevalences : int
        Number of prevalence points. For ``strategy='grid'`` this is the number
        of grid points *per class dimension*; for the sampling strategies it is
        the number of prevalence vectors drawn from the simplex.
    repeats : int, default=1
        Number of repetitions for each prevalence combination.
    random_state : int or None, default=None
        Random seed for reproducibility.
    min_prev : float, default=0.0
        Minimum class prevalence.
    max_prev : float, default=1.0
        Maximum class prevalence.
    strategy : {'grid', 'kraemer', 'uniform', 'dirichlet'}, default='grid'
        How prevalence vectors are generated over the simplex:

        - ``'grid'`` — a regular lattice of evenly-spaced prevalences from
          ``min_prev`` to ``max_prev`` (the classic APP). Deterministic and
          gives systematic coverage, but the number of points grows
          combinatorially (:math:`O(n^{k-1})` for ``k`` classes), so it is best
          for the binary / low-class-count case.
        - ``'kraemer'`` — the Kraemer method for *uniform* sampling over the
          simplex. Every prevalence combination is equally likely and the cost
          is independent of the number of classes, making it well suited to
          multiclass problems where a grid would explode.
        - ``'uniform'`` — uniform sampling over the simplex via the flat
          Dirichlet distribution :math:`\mathrm{Dir}(\mathbf{1})`. Statistically
          equivalent to ``'kraemer'`` (uniform coverage) but produced through
          the Dirichlet route; it is exactly ``'dirichlet'`` with
          ``dirichlet_alpha=1``.
        - ``'dirichlet'`` — sampling from a Dirichlet distribution whose
          concentration is set by ``dirichlet_alpha``. Use this to *bias* the
          sampling: ``alpha > 1`` favours balanced prevalences near the centre
          of the simplex, while ``alpha < 1`` favours extreme,
          one-class-dominant prevalences near the corners.
    dirichlet_alpha : float or array-like, default=1.0
        Concentration parameter used when ``strategy='dirichlet'``. A scalar is
        broadcast to a symmetric Dirichlet over all classes; an array of length
        ``n_classes`` sets a per-class concentration. Ignored by the other
        strategies (``'uniform'`` always uses ``1.0``).

    Attributes
    ----------
    n_combinations : int
        Total number of batches generated.

    Notes
    -----
    For multiclass problems the ``'grid'`` strategy grows combinatorially;
    prefer a sampling strategy (or :class:`UPP`) for large class counts.

    Examples
    --------
    >>> from mlquantify.model_selection import APP
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = APP(batch_size=50, n_prevalences=5, random_state=0)
    >>> batches = list(proto.split(X, y))
    >>> len(batches)
    5

    References
    ----------
    .. dropdown:: References

        .. [1] Forman, G. (2008). Quantifying Counts and Costs via Classification.
               *Data Mining and Knowledge Discovery*, 17(2), 164–206.
        .. [2] Sebastiani, F., et al. (2020). A Critical Reassessment of the
               Evaluation of Machine Learning Approaches for Quantification.
               *ArXiv preprint*.
    """

    _parameter_constraints = {
        "n_prevalences": [Interval(left=1, right=None, discrete=True)],
        "repeats": [Interval(left=1, right=None, discrete=True)],
        "min_prev": [Interval(left=0.0, right=1.0)],
        "max_prev": [Interval(left=0.0, right=1.0)],
        "strategy": [Options(['grid', 'kraemer', 'uniform', 'dirichlet'])],
    }

    def __init__(self, batch_size, n_prevalences, repeats=1, random_state=None,
                 min_prev=0.0, max_prev=1.0, strategy='grid', dirichlet_alpha=1.0):
        super().__init__(batch_size=batch_size,
                            random_state=random_state,
                            n_prevalences=n_prevalences,
                            repeats=repeats)
        self.min_prev = min_prev
        self.max_prev = max_prev
        self.strategy = strategy
        self.dirichlet_alpha = dirichlet_alpha

    def _sample_prevalences(self, n_dim, rng):
        """Generate prevalence vectors for one batch according to ``strategy``."""
        if self.strategy == 'grid':
            return simplex_grid_sampling(n_dim=n_dim,
                                         n_prev=self.n_prevalences,
                                         n_iter=self.repeats,
                                         min_val=self.min_prev,
                                         max_val=self.max_prev)
        if self.strategy == 'kraemer':
            return simplex_uniform_kraemer(n_dim=n_dim,
                                           n_prev=self.n_prevalences,
                                           n_iter=self.repeats,
                                           min_val=self.min_prev,
                                           max_val=self.max_prev,
                                           random_state=rng)
        # 'uniform' is the flat Dirichlet; 'dirichlet' uses dirichlet_alpha.
        alpha = 1.0 if self.strategy == 'uniform' else self.dirichlet_alpha
        return simplex_dirichlet_sampling(n_dim=n_dim,
                                          n_prev=self.n_prevalences,
                                          n_iter=self.repeats,
                                          alpha=alpha,
                                          min_val=self.min_prev,
                                          max_val=self.max_prev,
                                          random_state=rng)

    def _iter_indices(self, X: np.ndarray, y: np.ndarray):

        n_dim = len(np.unique(y))

        rng = check_random_state(self.random_state)

        for batch_size in self.batch_size:
            prevalences = self._sample_prevalences(n_dim, rng)
            for prev in prevalences:
                indexes = get_indexes_with_prevalence(y, prev, batch_size, random_state=rng)
                yield indexes

            

class NPP(BaseProtocol):
    r"""Natural Prevalence Protocol (NPP).

    Samples evaluation batches uniformly at random from the dataset, preserving
    the natural class distribution without imposing any prevalence constraints.

    Parameters
    ----------
    batch_size : int or list of int
        Size(s) of the evaluation batches.
    n_samples : int, default=1
        Number of distinct batch samples per batch size.
    repeats : int, default=1
        Number of repetitions for each sample.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    n_combinations : int
        Total number of batches generated.

    Examples
    --------
    >>> from mlquantify.model_selection import NPP
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = NPP(batch_size=50, n_samples=3, random_state=0)
    >>> batches = list(proto.split(X, y))
    >>> len(batches)
    3
    """
    
    _parameter_constraints = {
        "repeats": [Interval(left=1, right=None, discrete=True)]
    }
    
    def __init__(self, batch_size, n_samples=1, repeats=1, random_state=None):
        super().__init__(batch_size=batch_size, 
                        random_state=random_state)
        self.n_samples = n_samples
        self.repeats = repeats

    def _iter_indices(self, X: np.ndarray, y: np.ndarray):
        rng = check_random_state(self.random_state)
        for _ in range(self.n_samples):
            for batch_size in self.batch_size:
                idx = rng.choice(X.shape[0], batch_size, replace=True)
                for _ in range(self.repeats):
                    yield idx
            

class UPP(APP):
    r"""Uniform Prevalence Protocol (UPP).

    A thin specialisation of :class:`APP` that samples prevalences *uniformly*
    over the probability simplex rather than on a regular grid, avoiding the
    combinatorial blow-up of a grid for multiclass problems. It is exactly
    :class:`APP` with the simplex sampling ``strategy`` pinned on (``'kraemer'``
    by default).

    Parameters
    ----------
    batch_size : int or list of int
        Batch size(s) for evaluation.
    n_prevalences : int
        Number of prevalence points to sample.
    repeats : int, default=1
        Number of repetitions for each prevalence point.
    random_state : int or None, default=None
        Random seed for reproducibility.
    min_prev : float, default=0.0
        Minimum class prevalence.
    max_prev : float, default=1.0
        Maximum class prevalence.
    strategy : {'kraemer', 'uniform', 'dirichlet'}, default='kraemer'
        Simplex sampling strategy, forwarded to :class:`APP`. ``'grid'`` is not
        meaningful for UPP. See :class:`APP` for a description of each strategy.
    dirichlet_alpha : float or array-like, default=1.0
        Concentration parameter used when ``strategy='dirichlet'``; see
        :class:`APP`.
    algorithm : {'kraemer', 'uniform'}, optional
        .. deprecated::
            Use ``strategy`` instead. When given, its value is used as
            ``strategy`` for backward compatibility.

    Attributes
    ----------
    n_combinations : int
        Total number of batches generated.

    Examples
    --------
    >>> from mlquantify.model_selection import UPP
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = UPP(batch_size=50, n_prevalences=5, random_state=0)
    >>> batches = list(proto.split(X, y))
    >>> len(batches)
    5
    """

    _parameter_constraints = {
        "n_prevalences": [Interval(left=1, right=None, discrete=True)],
        "repeats": [Interval(left=1, right=None, discrete=True)],
        "min_prev": [Interval(left=0.0, right=1.0)],
        "max_prev": [Interval(left=0.0, right=1.0)],
        "strategy": [Options(['kraemer', 'uniform', 'dirichlet'])],
    }

    def __init__(self,
                 batch_size,
                 n_prevalences,
                 repeats=1,
                 random_state=None,
                 min_prev=0.0,
                 max_prev=1.0,
                 strategy='kraemer',
                 dirichlet_alpha=1.0,
                 algorithm=None):
        if algorithm is not None:
            warning(
                "The 'algorithm' parameter of UPP is deprecated; use 'strategy' "
                "instead."
            )
            strategy = algorithm
        super().__init__(batch_size=batch_size,
                         n_prevalences=n_prevalences,
                         repeats=repeats,
                         random_state=random_state,
                         min_prev=min_prev,
                         max_prev=max_prev,
                         strategy=strategy,
                         dirichlet_alpha=dirichlet_alpha)


class PPP(BaseProtocol):
    r"""Personalized Prevalence Protocol (PPP).

    Generates evaluation batches with explicitly specified class prevalences,
    enabling controlled evaluation at exact target operating points.

    Parameters
    ----------
    batch_size : int or list of int
        Batch sizes to generate.
    prevalences : list of float or array-like
        Target class prevalences. A single float is interpreted as the positive
        class prevalence in binary problems (negative = 1 - float).
    repeats : int, default=1
        Number of repetitions per prevalence point.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    n_combinations : int
        Total number of batches generated.

    Examples
    --------
    >>> from mlquantify.model_selection import PPP
    >>> import numpy as np
    >>> X, y = np.random.randn(200, 5), np.random.randint(0, 2, 200)
    >>> proto = PPP(batch_size=50, prevalences=[[0.2, 0.8], [0.5, 0.5]], random_state=0)
    >>> batches = list(proto.split(X, y))
    >>> len(batches)
    2
    """
    
    _parameter_constraints = {
        "repeats": [Interval(left=1, right=None, discrete=True)],
        "prevalences": ["array-like"]
    }
    
    def __init__(self, batch_size, prevalences, repeats=1, random_state=None):
        super().__init__(batch_size=batch_size, 
                        random_state=random_state,
                        prevalences=prevalences, 
                        repeats=repeats)
    
    def _iter_indices(self, X: np.ndarray, y: np.ndarray):
        rng = check_random_state(self.random_state)
        for batch_size in self.batch_size:    
            for prev in self.prevalences:
                if isinstance(prev, float):
                    prev = [1-prev, prev]

                indexes = get_indexes_with_prevalence(y, prev, batch_size, random_state=rng)
                yield indexes


# ===========================================
# High-level protocol runner
# ===========================================


def _resolve_scoring(scoring):
    """Resolve ``scoring`` into an ordered ``{name: callable}`` mapping."""
    from mlquantify import metrics as _metrics

    registry = {}
    for _name in ("AE", "SE", "MAE", "MSE", "KLD", "RAE", "NAE", "NRAE",
                  "NKLD", "NMD", "RNOD", "VSE", "CvM_L1"):
        if hasattr(_metrics, _name):
            registry[_name.lower()] = getattr(_metrics, _name)

    if scoring is None:
        scoring = ["mae"]
    if isinstance(scoring, str) or callable(scoring):
        scoring = [scoring]

    resolved = {}
    for item in scoring:
        if isinstance(item, str):
            key = item.lower()
            if key not in registry:
                raise ValueError(
                    f"Unknown scoring {item!r}. Available metrics: "
                    f"{sorted(registry)}."
                )
            fn = registry[key]
            resolved[fn.__name__] = fn
        elif callable(item):
            resolved[getattr(item, "__name__", "score")] = item
        else:
            raise TypeError(
                "scoring must be a metric name, a callable, or a list of those."
            )
    return resolved


def _build_protocol(protocol, batch_size, n_prevalences, repeats,
                    random_state, params):
    """Instantiate a protocol from a name, or pass through an instance."""
    if isinstance(protocol, BaseProtocol):
        return protocol
    if not isinstance(protocol, str):
        raise TypeError(
            "protocol must be a string ('app', 'npp', 'upp', 'ppp') or a "
            "BaseProtocol instance."
        )

    name = protocol.lower()
    if name == "app":
        return APP(batch_size=batch_size, n_prevalences=n_prevalences,
                   repeats=repeats, random_state=random_state,
                   **{k: params[k] for k in
                      ("min_prev", "max_prev", "strategy", "dirichlet_alpha")
                      if k in params})
    if name == "upp":
        return UPP(batch_size=batch_size, n_prevalences=n_prevalences,
                   repeats=repeats, random_state=random_state,
                   **{k: params[k] for k in
                      ("min_prev", "max_prev", "strategy", "dirichlet_alpha",
                       "algorithm")
                      if k in params})
    if name == "npp":
        return NPP(batch_size=batch_size,
                   n_samples=params.get("n_samples", n_prevalences),
                   repeats=repeats, random_state=random_state)
    if name == "ppp":
        if "prevalences" not in params:
            raise ValueError(
                "protocol='ppp' requires a `prevalences` keyword argument."
            )
        return PPP(batch_size=batch_size, prevalences=params["prevalences"],
                   repeats=repeats, random_state=random_state)

    raise ValueError(
        f"Unknown protocol {protocol!r}. Use 'app', 'npp', 'upp', 'ppp' or a "
        "BaseProtocol instance."
    )


def _prevalence_to_array(prevalence, classes):
    """Coerce a prediction (dict or array) to an array aligned to ``classes``."""
    if isinstance(prevalence, dict):
        return np.asarray([float(prevalence.get(c, 0.0)) for c in classes],
                          dtype=float)
    return np.asarray(prevalence, dtype=float)


def apply_protocol(
    quantifier,
    X,
    y,
    protocol="app",
    *,
    scoring="mae",
    batch_size=100,
    n_prevalences=11,
    repeats=1,
    fit=True,
    test_size=0.4,
    stratify=True,
    n_jobs=1,
    random_state=None,
    return_predictions=True,
    return_estimator=False,
    verbose=False,
    **protocol_params,
):
    r"""Evaluate a quantifier across an evaluation protocol.

    The protocol analogue of scikit-learn's :func:`~sklearn.model_selection.cross_validate`:
    it fits the quantifier, generates many test samples with controlled
    prevalences using a sampling protocol (:class:`APP`, :class:`NPP`,
    :class:`UPP` or :class:`PPP`), predicts the prevalence of each sample, and
    returns the true and predicted prevalences together with one score array per
    metric. This packages the standard quantification evaluation loop into a
    single call.

    Parameters
    ----------
    quantifier : BaseQuantifier
        The quantifier to evaluate. When ``fit=True`` a copy is trained, so the
        passed object is left untouched; when ``fit=False`` it must already be
        fitted.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Class labels.
    protocol : {'app', 'npp', 'upp', 'ppp'} or BaseProtocol, default='app'
        Sampling protocol used to build the evaluation samples.

        - ``'app'`` : Artificial Prevalence Protocol (grid of prevalences).
        - ``'npp'`` : Natural Prevalence Protocol (random natural samples).
        - ``'upp'`` : Uniform Prevalence Protocol (uniform over the simplex).
        - ``'ppp'`` : Personalized protocol (requires ``prevalences=...``).

        A pre-built :class:`BaseProtocol` instance may also be passed, in which
        case ``batch_size``, ``n_prevalences`` and ``repeats`` are ignored.
    scoring : str, callable, or list, default='mae'
        Metric(s) used to score each sample. A metric name (e.g. ``'mae'``,
        ``'nmd'``), a callable ``metric(pred, true) -> float``, or a list mixing
        the two. Each becomes a key in the returned dictionary.
    batch_size : int or list of int, default=100
        Size of each evaluation sample.
    n_prevalences : int, default=11
        Number of prevalence points (APP/UPP) or natural samples (NPP).
    repeats : int, default=1
        Number of repetitions per prevalence point.
    fit : bool, default=True
        If ``True``, train a copy of ``quantifier`` before evaluating; if
        ``False``, use the already-fitted ``quantifier`` directly.
    test_size : float or int or None, default=0.4
        When ``fit=True``, the held-out fraction (or count) the protocol samples
        from; the quantifier is trained on the complement. ``None`` or ``0``
        trains and evaluates on the same data (in-sample).
    stratify : bool, default=True
        Whether the train/evaluation split is stratified by ``y``.
    n_jobs : int or None, default=1
        Number of parallel jobs over the protocol samples.
    random_state : int, RandomState instance, or None, default=None
        Seed controlling the split and the protocol sampling.
    return_predictions : bool, default=True
        Whether to include the true and predicted prevalence arrays.
    return_estimator : bool, default=False
        Whether to include the fitted quantifier under the ``'estimator'`` key.
    verbose : bool, default=False
        Print the number of samples generated.
    **protocol_params : dict
        Extra protocol arguments forwarded to the constructor (e.g.
        ``min_prev``, ``max_prev``, ``strategy`` and ``dirichlet_alpha`` for
        APP/UPP, ``prevalences`` for PPP).

    Returns
    -------
    results : dict
        Dictionary with the keys:

        - ``'true_prevalences'`` : ndarray of shape (n_samples, n_classes),
          present when ``return_predictions=True``.
        - ``'predicted_prevalences'`` : ndarray of the same shape.
        - ``'n_batches'`` : int, the number of evaluation samples.
        - one key per metric (e.g. ``'MAE'``) : ndarray of shape (n_samples,)
          holding the per-sample score.
        - ``'estimator'`` : the fitted quantifier, when ``return_estimator=True``.

    See Also
    --------
    APP, NPP, UPP, PPP : The sampling protocols this runs.
    GridSearchQ : Hyper-parameter search using the same protocols.

    Notes
    -----
    Aggregate a run with ``results['MAE'].mean()``. The true and predicted
    prevalence arrays are convenient for diagonal "true vs predicted" diagnostic
    plots.

    Examples
    --------
    >>> from mlquantify.model_selection import apply_protocol
    >>> from mlquantify.counting import CC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=400, random_state=0)
    >>> results = apply_protocol(
    ...     CC(LogisticRegression()), X, y,
    ...     protocol="app", n_prevalences=11, batch_size=100,
    ...     scoring=["mae", "nmd"], random_state=0,
    ... )
    >>> results["true_prevalences"].shape       # doctest: +SKIP
    (11, 2)
    >>> round(float(results["MAE"].mean()), 2)   # doctest: +SKIP
    """
    X = np.asarray(X)
    y = np.asarray(y)

    scorers = _resolve_scoring(scoring)

    if fit:
        estimator = deepcopy(quantifier)
        if test_size:
            X_train, X_pool, y_train, y_pool = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=y if stratify else None,
            )
        else:
            X_train, y_train, X_pool, y_pool = X, y, X, y
        estimator.fit(X_train, y_train)
    else:
        estimator = quantifier
        X_pool, y_pool = X, y

    classes = getattr(estimator, "classes_", None)
    if classes is None:
        classes = np.unique(y_pool)
    classes = list(classes)

    proto = _build_protocol(protocol, batch_size, n_prevalences, repeats,
                            random_state, protocol_params)

    batches = list(proto.split(X_pool, y_pool))
    if verbose:
        print(f"[apply_protocol] {proto.__class__.__name__}: "
              f"{len(batches)} samples")

    def _evaluate(idx):
        X_batch, y_batch = X_pool[idx], y_pool[idx]
        true = get_prev_from_labels(y_batch, format="array", classes=classes)
        pred = _prevalence_to_array(estimator.predict(X_batch), classes)
        return true, pred

    outcomes = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate)(idx) for idx in batches
    )
    true_prevalences = np.asarray([t for t, _ in outcomes], dtype=float)
    predicted_prevalences = np.asarray([p for _, p in outcomes], dtype=float)

    results = {}
    if return_predictions:
        results["true_prevalences"] = true_prevalences
        results["predicted_prevalences"] = predicted_prevalences
    results["n_batches"] = len(batches)

    for name, fn in scorers.items():
        results[name] = np.asarray([
            fn(pred, true)
            for true, pred in zip(true_prevalences, predicted_prevalences)
        ])

    if return_estimator:
        results["estimator"] = estimator

    return results
        
