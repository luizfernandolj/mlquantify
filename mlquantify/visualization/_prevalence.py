"""Per-class prevalence bar chart (single-sample display)."""

import numpy as np

from ._base import (
    _check_ax,
    _default_class_names,
    _to_prevalence_array,
    _validate_style_kwargs,
)


class PrevalenceDisplay:
    """Bar chart of a single sample's predicted class prevalence.

    For one test sample, draw the predicted prevalence of each class as a bar,
    optionally next to the true prevalence and/or with error bars (e.g. from a
    confidence interval). This is the natural way to inspect a *single*
    quantifier prediction, especially in the multiclass setting.

    Parameters
    ----------
    predicted_prevalence : array-like of shape (n_classes,) or dict
        Predicted prevalence. Dicts are coerced using ``class_names`` ordering.
    true_prevalence : array-like of shape (n_classes,) or dict, default=None
        Optional ground-truth prevalence drawn alongside the prediction.
    class_names : list of str, default=None
        Class labels in vector order. Inferred from dict keys when available.
    yerr : array-like, default=None
        Error-bar sizes for the predicted bars, in the format accepted by
        ``ax.bar(yerr=...)`` (shape ``(n_classes,)`` or ``(2, n_classes)``).

    Attributes
    ----------
    bar_ : matplotlib BarContainer
        The predicted-prevalence bars.
    true_bar_ : matplotlib BarContainer or None
        The true-prevalence bars (None when ``true_prevalence`` is not given).
    ax_ : matplotlib Axes
    figure_ : matplotlib Figure

    See Also
    --------
    ConfidenceRegionDisplay : Uncertainty region for a single prediction.

    Examples
    --------
    >>> from mlquantify.visualization import PrevalenceDisplay
    >>> disp = PrevalenceDisplay.from_predictions(   # doctest: +SKIP
    ...     [0.3, 0.7], true_prevalence=[0.4, 0.6], class_names=["neg", "pos"])
    """

    def __init__(
        self,
        predicted_prevalence,
        *,
        true_prevalence=None,
        class_names=None,
        yerr=None,
    ):
        if class_names is None and isinstance(predicted_prevalence, dict):
            class_names = list(predicted_prevalence.keys())
        self.class_names = class_names
        self.predicted_prevalence = _to_prevalence_array(
            predicted_prevalence, class_names
        )
        self.true_prevalence = (
            None if true_prevalence is None
            else _to_prevalence_array(true_prevalence, class_names)
        )
        self.yerr = None if yerr is None else np.asarray(yerr, dtype=float)

    def plot(self, ax=None, *, name=None, **kwargs):
        """Draw the prevalence bars.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes to draw on.
        name : str, default=None
            Legend label for the predicted bars (defaults to ``"predicted"``).
        **kwargs
            Forwarded to the predicted-bar ``ax.bar`` call.

        Returns
        -------
        display : PrevalenceDisplay
        """
        fig, ax = _check_ax(ax)
        n_classes = self.predicted_prevalence.shape[0]
        class_names = _default_class_names(self.class_names, n_classes)
        x = np.arange(n_classes)

        paired = self.true_prevalence is not None
        width = 0.4 if paired else 0.6
        offset = width / 2 if paired else 0.0

        bar_kw = _validate_style_kwargs({"capsize": 4}, kwargs)
        self.bar_ = ax.bar(
            x - offset, self.predicted_prevalence, width,
            yerr=self.yerr, label=name or "predicted", **bar_kw,
        )
        self.true_bar_ = None
        if paired:
            self.true_bar_ = ax.bar(
                x + offset, self.true_prevalence, width, label="true", alpha=0.7
            )

        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_xlabel("Class")
        ax.set_ylabel("Prevalence")
        ax.set_ylim(0, 1)
        ax.legend(loc="best")

        self.ax_ = ax
        self.figure_ = fig
        return self

    @classmethod
    def from_predictions(
        cls,
        predicted_prevalence,
        *,
        true_prevalence=None,
        class_names=None,
        yerr=None,
        ax=None,
        **kwargs,
    ):
        """Build a :class:`PrevalenceDisplay` from a prevalence vector."""
        return cls(
            predicted_prevalence, true_prevalence=true_prevalence,
            class_names=class_names, yerr=yerr,
        ).plot(ax=ax, **kwargs)

    @classmethod
    def from_estimator(
        cls,
        quantifier,
        X,
        *,
        true_prevalence=None,
        ax=None,
        name=None,
        **kwargs,
    ):
        """Predict on ``X`` with ``quantifier`` and plot the prevalence.

        Parameters
        ----------
        quantifier : BaseQuantifier
            A fitted quantifier exposing ``predict`` and ``classes_``.
        X : array-like
            The single test sample to quantify.
        true_prevalence : array-like or dict, default=None
            Optional ground truth to draw alongside.
        ax : matplotlib Axes, default=None
        name : str, default=None
            Legend label for the predicted bars.
        **kwargs
            Passed to :meth:`plot`.

        Returns
        -------
        display : PrevalenceDisplay
        """
        class_names = getattr(quantifier, "classes_", None)
        prediction = quantifier.predict(X)
        return cls(
            prediction, true_prevalence=true_prevalence, class_names=class_names,
        ).plot(ax=ax, name=name, **kwargs)
