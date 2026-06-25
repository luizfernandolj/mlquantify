"""Error-by-shift curve (multiple-sample diagnostic)."""

import numpy as np

from ._base import _check_ax, _validate_style_kwargs


def _per_sample_error(true, pred, metric, eps=0.0):
    """Per-sample error array for ``true``/``pred`` of shape (n_samples, n_classes).

    For relative metrics (RAE) ``eps`` applies additive (Forman) smoothing,
    ``(p + eps) / (1 + n_classes * eps)``, so that zero-prevalence classes do
    not produce a division by zero (Sebastiani, 2020).
    """
    metric = metric.lower()
    if metric in ("ae", "mae"):
        return np.abs(pred - true).mean(axis=1)
    if metric in ("se", "mse"):
        return ((pred - true) ** 2).mean(axis=1)
    if metric in ("rae", "nrae"):
        from mlquantify.metrics import RAE

        n = true.shape[1]
        t = (true + eps) / (1 + n * eps)
        p = (pred + eps) / (1 + n * eps)
        return np.array([float(RAE(t[i], p[i])) for i in range(len(t))])
    raise ValueError(
        f"Unknown error_metric {metric!r}; expected one of "
        "'ae', 'se', 'rae'."
    )


class ErrorByShiftDisplay:
    """Estimation error as a function of prior-probability shift.

    For each evaluation sample, the *amount of shift* is the absolute-error
    distance between the sample's true prevalence and a reference (training)
    prevalence; the *error* is a quantification metric between the predicted and
    true prevalence. Samples are grouped into shift bins and the mean error per
    bin is drawn as a curve, optionally with a ``±std`` band. This reveals how a
    quantifier degrades as the test distribution drifts away from training — a
    standard robustness diagnostic in the learning-to-quantify literature.

    This is a *multiple-sample* display.

    Parameters
    ----------
    true_prevalences : ndarray of shape (n_samples, n_classes)
        True prevalence of each evaluation sample.
    predicted_prevalences : ndarray of shape (n_samples, n_classes)
        Predicted prevalence of each evaluation sample.
    train_prevalence : ndarray of shape (n_classes,), default=None
        Reference prevalence against which shift is measured. When None, the
        mean true prevalence across samples is used.
    error_metric : {'ae', 'se', 'rae'}, default='ae'
        Per-sample error measure plotted on the y-axis.
    smoothing : float, default=None
        Additive (Forman) smoothing ``eps`` applied before computing RAE, to
        avoid division by zero on zero-prevalence classes. When None, a small
        default of ``1e-3`` is used (``from_protocol`` overrides this with
        ``1 / (2 * batch_size)``). Ignored by non-relative metrics.

    Attributes
    ----------
    shift_ : ndarray of shape (n_samples,)
        Per-sample shift magnitude.
    error_ : ndarray of shape (n_samples,)
        Per-sample error.
    line_ : matplotlib Line2D
        The mean-error curve.
    fill_ : matplotlib PolyCollection or None
        The ``±std`` band (None when ``show_std=False``).
    ax_ : matplotlib Axes
    figure_ : matplotlib Figure

    Examples
    --------
    >>> from mlquantify.visualization import ErrorByShiftDisplay
    >>> from mlquantify.counting import CC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=400, random_state=0)
    >>> disp = ErrorByShiftDisplay.from_protocol(   # doctest: +SKIP
    ...     CC(LogisticRegression()), X, y, protocol="upp", error_metric="ae")
    """

    def __init__(
        self,
        true_prevalences,
        predicted_prevalences,
        *,
        train_prevalence=None,
        error_metric="ae",
        smoothing=None,
    ):
        self.true_prevalences = np.asarray(true_prevalences, dtype=float)
        self.predicted_prevalences = np.asarray(predicted_prevalences, dtype=float)
        self.train_prevalence = (
            None if train_prevalence is None
            else np.asarray(train_prevalence, dtype=float)
        )
        self.error_metric = error_metric
        self.smoothing = smoothing

    def plot(self, ax=None, *, n_bins=10, name=None, show_std=True, **kwargs):
        """Draw the error-by-shift curve.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes to draw on.
        n_bins : int, default=10
            Number of equal-width shift bins.
        name : str, default=None
            Legend label for the curve.
        show_std : bool, default=True
            Whether to draw a ``±std`` shaded band.
        **kwargs
            Forwarded to ``ax.plot`` (the mean-error curve).

        Returns
        -------
        display : ErrorByShiftDisplay
        """
        fig, ax = _check_ax(ax)

        ref = (
            self.true_prevalences.mean(axis=0)
            if self.train_prevalence is None
            else self.train_prevalence
        )
        self.shift_ = np.abs(self.true_prevalences - ref).mean(axis=1)
        eps = self.smoothing if self.smoothing is not None else 1e-3
        self.error_ = _per_sample_error(
            self.true_prevalences, self.predicted_prevalences,
            self.error_metric, eps=eps,
        )

        smax = self.shift_.max()
        edges = np.linspace(0.0, smax if smax > 0 else 1.0, n_bins + 1)
        idx = np.clip(np.digitize(self.shift_, edges[1:-1]), 0, n_bins - 1)
        centers, means, stds = [], [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.any():
                centers.append((edges[b] + edges[b + 1]) / 2)
                means.append(self.error_[mask].mean())
                stds.append(self.error_[mask].std())
        centers, means, stds = map(np.asarray, (centers, means, stds))

        line_kw = _validate_style_kwargs({"marker": "o"}, kwargs)
        if name is not None:
            line_kw["label"] = name
        (self.line_,) = ax.plot(centers, means, **line_kw)

        self.fill_ = None
        if show_std and len(centers):
            self.fill_ = ax.fill_between(
                centers, means - stds, means + stds,
                alpha=0.2, color=self.line_.get_color(),
            )

        ax.set_xlabel("Prior-probability shift (AE from reference prevalence)")
        ax.set_ylabel(f"{self.error_metric.upper()}")
        if name is not None:
            ax.legend(loc="best")

        self.ax_ = ax
        self.figure_ = fig
        return self

    @classmethod
    def from_predictions(
        cls,
        true_prevalences,
        predicted_prevalences,
        *,
        train_prevalence=None,
        error_metric="ae",
        smoothing=None,
        ax=None,
        **kwargs,
    ):
        """Build an :class:`ErrorByShiftDisplay` from precomputed arrays."""
        return cls(
            true_prevalences, predicted_prevalences,
            train_prevalence=train_prevalence, error_metric=error_metric,
            smoothing=smoothing,
        ).plot(ax=ax, **kwargs)

    @classmethod
    def from_protocol(
        cls,
        quantifier,
        X,
        y,
        *,
        protocol="upp",
        error_metric="ae",
        ax=None,
        name=None,
        n_bins=10,
        show_std=True,
        **protocol_kwargs,
    ):
        """Run a protocol and plot error against prior-probability shift.

        The reference (training) prevalence is taken from the class proportions
        of ``y``. ``**protocol_kwargs`` are forwarded to
        :func:`mlquantify.model_selection.apply_protocol`. The uniform protocol
        (``'upp'``) is the default because it spreads samples across the shift
        range.
        """
        from mlquantify.model_selection import apply_protocol

        results = apply_protocol(
            quantifier, X, y, protocol=protocol,
            return_predictions=True, **protocol_kwargs,
        )
        classes, counts = np.unique(y, return_counts=True)
        train_prevalence = counts / counts.sum()
        batch_size = protocol_kwargs.get("batch_size", 100)
        if isinstance(batch_size, (list, tuple, np.ndarray)):
            batch_size = int(np.min(batch_size))
        smoothing = 1.0 / (2 * batch_size)
        return cls(
            results["true_prevalences"],
            results["predicted_prevalences"],
            train_prevalence=train_prevalence,
            error_metric=error_metric,
            smoothing=smoothing,
        ).plot(ax=ax, name=name, n_bins=n_bins, show_std=show_std)
