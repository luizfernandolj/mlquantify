"""Signed-error bias boxplots (multiple-sample diagnostic)."""

import numpy as np

from ._base import _check_ax, _default_class_names


class BiasDisplay:
    """Boxplots of signed prevalence-estimation error.

    Visualises the *signed* error ``predicted - true`` across the samples of an
    evaluation protocol. A box centred above zero reveals systematic
    over-estimation; below zero, under-estimation; a tall box reveals high
    variance. Two layouts are available:

    - **global** (default): one box per class.
    - **binned**: for a single class, one box per bin of the true prevalence,
      exposing how the bias changes along the prevalence range.

    This is a *multiple-sample* display. It follows the boxplot convention used
    throughout the quantification literature for reporting estimation error
    (e.g. González et al., 2024).

    Parameters
    ----------
    true_prevalences : ndarray of shape (n_samples, n_classes)
        True prevalence of each evaluation sample.
    predicted_prevalences : ndarray of shape (n_samples, n_classes)
        Predicted prevalence of each evaluation sample.
    class_names : list of str, default=None
        Class labels in column order.

    Attributes
    ----------
    boxplot_ : dict
        The dictionary returned by ``ax.boxplot``.
    hline_ : matplotlib Line2D
        The horizontal zero-bias reference line.
    ax_ : matplotlib Axes
    figure_ : matplotlib Figure

    See Also
    --------
    DiagonalDisplay : True vs. predicted prevalence scatter.

    Examples
    --------
    >>> from mlquantify.visualization import BiasDisplay
    >>> from mlquantify.counting import CC
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=400, random_state=0)
    >>> disp = BiasDisplay.from_protocol(   # doctest: +SKIP
    ...     CC(LogisticRegression()), X, y, bins=5)
    """

    def __init__(self, true_prevalences, predicted_prevalences, *, class_names=None):
        self.true_prevalences = np.asarray(true_prevalences, dtype=float)
        self.predicted_prevalences = np.asarray(predicted_prevalences, dtype=float)
        self.class_names = class_names

    def plot(
        self,
        ax=None,
        *,
        bins=None,
        class_index=None,
        name=None,
        **kwargs,
    ):
        """Draw the bias boxplots.

        Parameters
        ----------
        ax : matplotlib Axes, default=None
            Axes to draw on.
        bins : int, default=None
            If given, draw the *binned* layout: the chosen class's samples are
            grouped into ``bins`` equal-width bins of true prevalence. If None,
            draw one box per class (global layout).
        class_index : int, default=None
            Class column used by the binned layout. Defaults to the last class
            (the conventional "positive" class).
        name : str, default=None
            Unused label kept for API symmetry; reserved for future legends.
        **kwargs
            Forwarded to ``ax.boxplot``.

        Returns
        -------
        display : BiasDisplay
        """
        fig, ax = _check_ax(ax)
        n_classes = self.true_prevalences.shape[1]
        class_names = _default_class_names(self.class_names, n_classes)
        error = self.predicted_prevalences - self.true_prevalences

        boxplot_kw = {"showfliers": True, "patch_artist": False}
        boxplot_kw.update(kwargs)

        if bins is None:
            data = [error[:, c] for c in range(n_classes)]
            positions = np.arange(1, n_classes + 1)
            self.boxplot_ = ax.boxplot(data, positions=positions, **boxplot_kw)
            ax.set_xticks(positions)
            ax.set_xticklabels(class_names)
            ax.set_xlabel("Class")
        else:
            if class_index is None:
                class_index = n_classes - 1
            true_c = self.true_prevalences[:, class_index]
            err_c = error[:, class_index]
            edges = np.linspace(0.0, 1.0, bins + 1)
            idx = np.clip(np.digitize(true_c, edges[1:-1]), 0, bins - 1)
            data, labels, positions = [], [], []
            for b in range(bins):
                mask = idx == b
                if mask.any():
                    data.append(err_c[mask])
                    labels.append(f"{edges[b]:.2f}-{edges[b + 1]:.2f}")
                    positions.append(len(positions) + 1)
            self.boxplot_ = ax.boxplot(data, positions=positions, **boxplot_kw)
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlabel(f"True prevalence of class '{class_names[class_index]}'")

        self.hline_ = ax.axhline(0.0, color="grey", linestyle="--", linewidth=1)
        ax.set_ylabel("Signed error (predicted - true)")

        self.ax_ = ax
        self.figure_ = fig
        return self

    @classmethod
    def from_predictions(
        cls,
        true_prevalences,
        predicted_prevalences,
        *,
        class_names=None,
        ax=None,
        **kwargs,
    ):
        """Build a :class:`BiasDisplay` from precomputed prevalence arrays."""
        return cls(
            true_prevalences, predicted_prevalences, class_names=class_names
        ).plot(ax=ax, **kwargs)

    @classmethod
    def from_protocol(
        cls,
        quantifier,
        X,
        y,
        *,
        protocol="app",
        ax=None,
        bins=None,
        class_index=None,
        name=None,
        **protocol_kwargs,
    ):
        """Run an evaluation protocol and plot the resulting bias boxplots.

        Wrapper around :func:`mlquantify.model_selection.apply_protocol`;
        ``**protocol_kwargs`` are forwarded to it.
        """
        from mlquantify.model_selection import apply_protocol

        results = apply_protocol(
            quantifier, X, y, protocol=protocol,
            return_predictions=True, **protocol_kwargs,
        )
        return cls(
            results["true_prevalences"],
            results["predicted_prevalences"],
            class_names=np.unique(y),
        ).plot(ax=ax, bins=bins, class_index=class_index, name=name)
