"""Confidence-region display for a single prevalence prediction."""

import numpy as np
from scipy.stats import chi2

from ._base import _check_ax, _default_class_names, _validate_style_kwargs


# 2-D coordinates of the 3-class probability simplex corners.
_TERNARY_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])


class ConfidenceRegionDisplay:
    """Confidence region around a single prevalence prediction.

    Visualises the uncertainty of one quantifier prediction, obtained from a
    set of bootstrap prevalence estimates (see :mod:`mlquantify.confidence`).
    The layout adapts to the number of classes:

    - **3 classes** (``kind='ternary'``): the bootstrap estimates are projected
      onto the 2-simplex and a confidence **ellipse** is drawn, together with
      the point estimate and (optionally) the true prevalence.
    - **otherwise** (``kind='interval'``): a per-class "forest" plot of the
      point estimate with a percentile confidence interval per class.

    This is a *single-sample* display.

    Parameters
    ----------
    prev_estims : ndarray of shape (m, n_classes)
        ``m`` bootstrap prevalence estimates for one prediction.
    confidence_level : float, default=0.95
        Confidence level of the region.
    class_names : list of str, default=None
        Class labels in column order.
    true_prevalence : array-like of shape (n_classes,), default=None
        Optional ground-truth prevalence to overlay.

    Attributes
    ----------
    point_estimate_ : ndarray of shape (n_classes,)
        Mean of the bootstrap estimates.
    ellipse_ : matplotlib Ellipse or None
        The confidence ellipse (ternary layout only).
    scatter_ : matplotlib PathCollection or None
        The projected bootstrap cloud (ternary layout only).
    errorbar_ : matplotlib ErrorbarContainer or None
        The per-class intervals (interval layout only).
    ax_ : matplotlib Axes
    figure_ : matplotlib Figure

    See Also
    --------
    mlquantify.confidence.construct_confidence_region : Builds the regions.
    PrevalenceDisplay : Plain per-class prevalence bars.

    Examples
    --------
    >>> import numpy as np
    >>> from mlquantify.visualization import ConfidenceRegionDisplay
    >>> estims = np.random.dirichlet([8, 6, 6], size=300)
    >>> disp = ConfidenceRegionDisplay.from_estimates(   # doctest: +SKIP
    ...     estims, confidence_level=0.95, class_names=["a", "b", "c"])
    """

    def __init__(
        self,
        prev_estims,
        *,
        confidence_level=0.95,
        class_names=None,
        true_prevalence=None,
    ):
        self.prev_estims = np.asarray(prev_estims, dtype=float)
        self.confidence_level = confidence_level
        self.class_names = class_names
        self.true_prevalence = (
            None if true_prevalence is None
            else np.asarray(true_prevalence, dtype=float)
        )

    def plot(self, ax=None, *, kind="auto", name=None, **kwargs):
        """Draw the confidence region.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes to draw on.
        kind : {'auto', 'ternary', 'interval'}, default='auto'
            Layout. ``'auto'`` selects ``'ternary'`` for 3-class problems and
            ``'interval'`` otherwise.
        name : str, default=None
            Label for the point estimate.
        **kwargs
            Forwarded to the primary artist (``ax.scatter`` of the bootstrap
            cloud for ternary; ``ax.errorbar`` for the interval layout).

        Returns
        -------
        display : ConfidenceRegionDisplay
        """
        fig, ax = _check_ax(ax)
        n_classes = self.prev_estims.shape[1]
        class_names = _default_class_names(self.class_names, n_classes)
        self.point_estimate_ = self.prev_estims.mean(axis=0)

        if kind == "auto":
            kind = "ternary" if n_classes == 3 else "interval"
        if kind == "ternary" and n_classes != 3:
            raise ValueError("kind='ternary' requires exactly 3 classes.")

        self.ellipse_ = self.scatter_ = self.errorbar_ = None
        if kind == "ternary":
            self._plot_ternary(ax, class_names, name, **kwargs)
        else:
            self._plot_interval(ax, class_names, name, **kwargs)

        self.ax_ = ax
        self.figure_ = fig
        return self

    def _plot_ternary(self, ax, class_names, name, **kwargs):
        from matplotlib.patches import Ellipse

        pts = self.prev_estims @ _TERNARY_CORNERS
        # Triangle edges + corner labels.
        tri = np.vstack([_TERNARY_CORNERS, _TERNARY_CORNERS[0]])
        ax.plot(tri[:, 0], tri[:, 1], color="grey", linewidth=1)
        for corner, label in zip(_TERNARY_CORNERS, class_names):
            ax.annotate(
                label, corner, ha="center", va="center",
                xytext=(corner - [0.5, 0.4 / np.sqrt(3)]) * 0.12 + corner,
            )

        scatter_kw = _validate_style_kwargs({"s": 10, "alpha": 0.3}, kwargs)
        self.scatter_ = ax.scatter(pts[:, 0], pts[:, 1], **scatter_kw)

        mean2d = pts.mean(axis=0)
        cov = np.cov(pts, rowvar=False)
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        scale = chi2.ppf(self.confidence_level, df=2)
        width, height = 2 * np.sqrt(np.maximum(vals, 0) * scale)
        self.ellipse_ = Ellipse(
            xy=mean2d, width=width, height=height, angle=angle,
            fill=False, edgecolor="C3", linewidth=2,
            label=f"{int(self.confidence_level * 100)}% region",
        )
        ax.add_patch(self.ellipse_)
        ax.plot(*mean2d, "o", color="C3", label=name or "estimate")

        if self.true_prevalence is not None:
            true2d = self.true_prevalence @ _TERNARY_CORNERS
            ax.plot(*true2d, "*", color="k", markersize=12, label="true")

        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="best")

    def _plot_interval(self, ax, class_names, name, **kwargs):
        alpha = 1 - self.confidence_level
        low, high = np.percentile(
            self.prev_estims, [alpha / 2 * 100, (1 - alpha / 2) * 100], axis=0
        )
        point = self.point_estimate_
        y = np.arange(len(class_names))
        err = np.vstack([point - low, high - point])

        eb_kw = _validate_style_kwargs({"fmt": "o", "capsize": 4}, kwargs)
        self.errorbar_ = ax.errorbar(
            point, y, xerr=err, label=name or "estimate", **eb_kw,
        )
        if self.true_prevalence is not None:
            ax.scatter(
                self.true_prevalence, y, marker="*", s=120, color="k",
                zorder=3, label="true",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(class_names)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Prevalence")
        ax.set_ylabel("Class")
        ax.legend(loc="best")

    @classmethod
    def from_estimates(
        cls,
        prev_estims,
        *,
        confidence_level=0.95,
        class_names=None,
        true_prevalence=None,
        ax=None,
        **kwargs,
    ):
        """Build a display directly from bootstrap prevalence estimates."""
        return cls(
            prev_estims, confidence_level=confidence_level,
            class_names=class_names, true_prevalence=true_prevalence,
        ).plot(ax=ax, **kwargs)

    @classmethod
    def from_region(
        cls,
        region,
        *,
        class_names=None,
        true_prevalence=None,
        ax=None,
        **kwargs,
    ):
        """Build a display from a fitted :mod:`mlquantify.confidence` region.

        Parameters
        ----------
        region : BaseConfidenceRegion
            A region instance exposing ``prev_estims`` and
            ``confidence_level`` (e.g. the output of
            :func:`mlquantify.confidence.construct_confidence_region`).
        class_names : list of str, default=None
        true_prevalence : array-like, default=None
        ax : matplotlib Axes, default=None
        **kwargs
            Passed to :meth:`plot`.

        Returns
        -------
        display : ConfidenceRegionDisplay
        """
        return cls(
            region.prev_estims,
            confidence_level=getattr(region, "confidence_level", 0.95),
            class_names=class_names,
            true_prevalence=true_prevalence,
        ).plot(ax=ax, **kwargs)
