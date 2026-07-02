import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from mlquantify.base import BaseQuantifier
from mlquantify.solvers import minimize_prevalence
from mlquantify.utils._constraints import Interval
from mlquantify.utils._validation import validate_data, validate_prevalences


def _pairwise_abs_diff(means):
    """Mean absolute pairwise difference between rows, per column.

    ``means`` has shape (n_classes, n_projections); the result has shape
    (n_pairs, n_projections) with one row per unordered class pair.
    """
    n = means.shape[0]
    rows, cols = torch.triu_indices(n, n, offset=1)
    return (means[rows] - means[cols]).abs()


class ReadMe2(BaseQuantifier):
    r"""ReadMe2 quantifier (Jerzak, King & Strezhnev, 2022).

    Targets prior probability shift and semantic change. The continuous-feature
    successor of :class:`ReadMe`: a **non-aggregative** quantifier that solves
    the accounting identity :math:`E[\tilde{S}] = E[\tilde{S} \mid D] P(D)` by
    least squares on the simplex, where :math:`\tilde{S}` are **learned
    projections** of the document features. Per bootstrap iteration the
    projection :math:`\Gamma` (a linear map followed by a softsign
    nonlinearity) is optimized by stochastic gradient descent on
    category-balanced labeled batches to maximise *category distinctiveness*
    (separation of the per-class projected means) and *feature distinctiveness*
    (diversity among the projections' discrimination patterns). Before
    estimation, labeled documents are matched to the unlabeled set by
    k-nearest neighbours, reducing bias when the language of a category
    drifts between the two sets.

    Requires PyTorch (``pip install mlquantify[neural]``) and continuous
    document features (e.g. word-embedding summaries).

    Parameters
    ----------
    n_boot : int, default=15
        Number of bootstrap iterations (independent projections, averaged).
    sgd_iters : int, default=500
        SGD steps per bootstrap iteration.
    n_projections : int or None, default=None
        Number of learned projections. ``None`` uses ``n_classes + 2``.
    batch_size_per_cat : int, default=10
        Labeled documents sampled per class in each SGD batch.
    k_match : int, default=3
        Neighbours per unlabeled document in the matching step.
    n_boot_match : int, default=50
        Matched-estimation rounds averaged within each bootstrap iteration.
    min_match : int, default=8
        Minimum matched documents per class; classes below it are topped up
        by resampling their labeled documents.
    matching : bool, default=True
        Whether to apply kNN matching before estimation. ``False`` estimates
        from all labeled documents (the reference's "no matching" variant).
    wt_cat_distinctiveness : float or None, default=None
        Weight of the category-distinctiveness term (``None`` = 1.0).
    wt_feat_distinctiveness : float or None, default=None
        Weight of the feature-distinctiveness term (``None`` = 1.0).
    learning_rate : float, default=0.01
        SGD learning rate (momentum 0.9, Nesterov).
    device : str or None, default=None
        Torch device; ``None`` selects CUDA when available, else CPU.
    random_state : int or None, default=None
        Seed for batching, matching, and torch initialisation.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.
    transforms_ : list of tuples
        The ``n_boot`` fitted projections ``(W, bias, mu, sigma)``.
    winsor_bounds_ : ndarray of shape (2, n_features)
        Per-feature Tukey fences used to clip inputs.

    Notes
    -----
    Like :class:`ReadMe`, the identifying assumption is the stability of the
    class-conditional feature distribution across the labeled and unlabeled
    sets; the matching step relaxes it further by re-weighting the labeled
    set toward the region of feature space actually occupied by the unlabeled
    documents. Estimates are averaged over ``n_boot`` independently learned
    projections, which stabilises the otherwise stochastic optimisation.

    See Also
    --------
    ReadMe : The original binary-profile method.

    Examples
    --------
    >>> from mlquantify.readme import ReadMe2
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=300, random_state=42)
    >>> q = ReadMe2(n_boot=2, sgd_iters=50, n_boot_match=5, random_state=0).fit(X, y)
    >>> q.predict(X)
    {0: ..., 1: ...}

    References
    ----------
    .. dropdown:: References

        .. [1] Jerzak, C. T., King, G., & Strezhnev, A. (2022). An Improved
               Method of Automated Nonparametric Content Analysis for Social
               Science. *Political Analysis*, 31(1), 42-58.
        .. [2] Hopkins, D. J., & King, G. (2010). A Method of Automated
               Nonparametric Content Analysis for Social Science.
               *American Journal of Political Science*, 54(1), 229-247.
    """

    _parameter_constraints = {
        "n_boot": [Interval(1, None, inclusive_right=False)],
        "sgd_iters": [Interval(1, None, inclusive_right=False)],
        "n_projections": [Interval(1, None, inclusive_right=False), type(None)],
        "batch_size_per_cat": [Interval(1, None, inclusive_right=False)],
        "k_match": [Interval(1, None, inclusive_right=False)],
        "n_boot_match": [Interval(1, None, inclusive_right=False)],
        "min_match": [Interval(1, None, inclusive_right=False)],
        "matching": [bool],
        "wt_cat_distinctiveness": [Interval(0.0, None), type(None)],
        "wt_feat_distinctiveness": [Interval(0.0, None), type(None)],
        "learning_rate": [Interval(0.0, None, inclusive_left=False)],
        "device": [str, type(None)],
        "random_state": [Interval(0, None, inclusive_right=False), type(None)],
    }

    def __init__(self,
                 n_boot=15,
                 sgd_iters=500,
                 n_projections=None,
                 batch_size_per_cat=10,
                 k_match=3,
                 n_boot_match=50,
                 min_match=8,
                 matching=True,
                 wt_cat_distinctiveness=None,
                 wt_feat_distinctiveness=None,
                 learning_rate=0.01,
                 device=None,
                 random_state=None):
        self.n_boot = n_boot
        self.sgd_iters = sgd_iters
        self.n_projections = n_projections
        self.batch_size_per_cat = batch_size_per_cat
        self.k_match = k_match
        self.n_boot_match = n_boot_match
        self.min_match = min_match
        self.matching = matching
        self.wt_cat_distinctiveness = wt_cat_distinctiveness
        self.wt_feat_distinctiveness = wt_feat_distinctiveness
        self.learning_rate = learning_rate
        self.device = device
        self.random_state = random_state

    def _resolve_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _balanced_batch(self, rng):
        return np.concatenate([
            rng.choice(idx, size=self.batch_size_per_cat, replace=True)
            for idx in self._class_indices
        ])

    def fit(self, X, y):
        r"""Winsorize the labeled data and learn the bootstrap projections.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Labeled continuous features.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self : ReadMe2
            Fitted quantifier.
        """
        self._validate_params()
        X, y = validate_data(self, X, y)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        n_projections = self.n_projections or n_classes + 2
        wt_cat = 1.0 if self.wt_cat_distinctiveness is None else self.wt_cat_distinctiveness
        wt_feat = 1.0 if self.wt_feat_distinctiveness is None else self.wt_feat_distinctiveness

        # Tukey fences per feature (port of Winsorize_values).
        q1, q3 = np.percentile(X, [25, 75], axis=0)
        fence = 1.5 * (q3 - q1)
        self.winsor_bounds_ = np.stack([q1 - fence, q3 + fence])
        X = np.clip(X, self.winsor_bounds_[0], self.winsor_bounds_[1])

        self._X_labeled = X
        self._y_idx = y_idx
        self._class_indices = [np.flatnonzero(y_idx == j) for j in range(n_classes)]

        rng = check_random_state(self.random_state)
        device = self._resolve_device()
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
        n_features = X.shape[1]
        batch_class = torch.as_tensor(
            np.repeat(np.arange(n_classes), self.batch_size_per_cat),
            device=device,
        )

        self.transforms_ = []
        for _ in range(self.n_boot):
            torch.manual_seed(int(rng.randint(np.iinfo(np.int32).max)))

            # Standardisation statistics from category-balanced batches, so
            # they are not dominated by the labeled class distribution.
            stat_batches = np.stack([self._balanced_batch(rng) for _ in range(100)])
            stat_data = X[stat_batches.ravel()]
            mu = stat_data.mean(axis=0)
            sigma = stat_data.std(axis=0) + 1e-6

            mu_t = torch.as_tensor(mu, dtype=torch.float32, device=device)
            sigma_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
            W = torch.empty(n_features, n_projections, device=device)
            torch.nn.init.xavier_uniform_(W)
            W.requires_grad_(True)
            bias = torch.zeros(n_projections, device=device, requires_grad=True)
            optimizer = torch.optim.SGD(
                [W, bias], lr=self.learning_rate, momentum=0.9, nesterov=True,
            )

            for _ in range(self.sgd_iters):
                batch = X_t[self._balanced_batch(rng)]
                z = (batch - mu_t) / sigma_t
                t = torch.nn.functional.softsign(z @ W + bias)
                # normalise within the batch so the objective terms are
                # scale-free across projections
                t = (t - t.mean(dim=0)) / (t.std(dim=0) + 1e-6)

                class_means = torch.stack([
                    t[batch_class == j].mean(dim=0) for j in range(n_classes)
                ])
                # (n_class_pairs, n_projections): each projection's
                # discrimination pattern across class pairs
                discrimination = _pairwise_abs_diff(class_means)
                cat_distinctiveness = discrimination.mean()
                if n_projections > 1:
                    feat_distinctiveness = _pairwise_abs_diff(discrimination.T).mean()
                else:
                    feat_distinctiveness = torch.zeros((), device=device)
                within_spread = torch.stack([
                    t[batch_class == j].std(dim=0) for j in range(n_classes)
                ])
                spread_term = torch.log(within_spread.min(dim=0).values + 1e-3).mean()

                loss = -(wt_cat * cat_distinctiveness
                         + wt_feat * feat_distinctiveness
                         + 0.01 * spread_term)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([W, bias], max_norm=5.0)
                optimizer.step()

            self.transforms_.append((
                W.detach().cpu().numpy(),
                bias.detach().cpu().numpy(),
                mu,
                sigma,
            ))
        return self

    def _apply_transform(self, X, transform):
        W, bias, mu, sigma = transform
        t = ((X - mu) / sigma) @ W + bias
        return t / (1.0 + np.abs(t))

    def _estimate(self, class_means, unlabeled_mean, n_classes):
        def objective(prevalences):
            residual = prevalences @ class_means - unlabeled_mean
            return float(residual @ residual)

        prevalences, _ = minimize_prevalence(objective, n_classes, solver="slsqp")
        return prevalences

    def predict(self, X):
        r"""Estimate class prevalences on the given (unlabeled) data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        ndarray of shape (n_classes,)
            Class prevalence estimates.
        """
        X = validate_data(self, X)
        X = np.clip(X, self.winsor_bounds_[0], self.winsor_bounds_[1])
        n_classes = len(self.classes_)
        rng = check_random_state(self.random_state)

        estimates = []
        for transform in self.transforms_:
            t_labeled = self._apply_transform(self._X_labeled, transform)
            t_unlabeled = self._apply_transform(X, transform)
            unlabeled_mean = t_unlabeled.mean(axis=0)

            if not self.matching:
                class_means = np.stack([
                    t_labeled[idx].mean(axis=0) for idx in self._class_indices
                ])
                estimates.append(self._estimate(class_means, unlabeled_mean, n_classes))
                continue

            nn = NearestNeighbors(n_neighbors=self.k_match).fit(t_labeled)
            round_estimates = []
            for _ in range(self.n_boot_match):
                query_size = min(len(t_unlabeled), 200)
                queries = rng.choice(len(t_unlabeled), size=query_size, replace=True)
                matched = nn.kneighbors(
                    t_unlabeled[queries], return_distance=False
                ).ravel()
                class_means = []
                for j, idx in enumerate(self._class_indices):
                    matched_j = matched[np.isin(matched, idx)]
                    if len(matched_j) < self.min_match:
                        top_up = rng.choice(
                            idx, size=self.min_match - len(matched_j), replace=True
                        )
                        matched_j = np.concatenate([matched_j, top_up])
                    class_means.append(t_labeled[matched_j].mean(axis=0))
                round_estimates.append(
                    self._estimate(np.stack(class_means), unlabeled_mean, n_classes)
                )
            estimates.append(np.mean(round_estimates, axis=0))

        prevalences = np.mean(estimates, axis=0)
        prevalences = prevalences / prevalences.sum()
        return validate_prevalences(self, prevalences, self.classes_)
