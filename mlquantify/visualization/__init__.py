"""Visualization utilities for quantification.

A small collection of matplotlib ``*Display`` classes, following the
scikit-learn Display API (``from_predictions`` / ``from_estimator`` /
``from_protocol`` constructors, a ``plot`` method, and stored ``ax_`` /
``figure_`` attributes). The plots are quantification-specific and do not
duplicate scikit-learn's classification/regression displays.

Multiple-sample diagnostics (summarise an evaluation protocol run):

- :class:`DiagonalDisplay` — true vs. predicted prevalence scatter.
- :class:`BiasDisplay` — signed-error boxplots, global or binned.
- :class:`ErrorByShiftDisplay` — error vs. prior-probability shift.

Single-sample displays (inspect one prediction):

- :class:`PrevalenceDisplay` — per-class predicted prevalence bars.
- :class:`ConfidenceRegionDisplay` — confidence interval / ternary ellipse.

This subpackage is intentionally not imported by ``import mlquantify`` so that
matplotlib stays off the top-level import path; import it explicitly::

    from mlquantify.visualization import DiagonalDisplay
"""

from ._diagonal import DiagonalDisplay
from ._bias import BiasDisplay
from ._shift import ErrorByShiftDisplay
from ._prevalence import PrevalenceDisplay
from ._confidence import ConfidenceRegionDisplay

__all__ = [
    "DiagonalDisplay",
    "BiasDisplay",
    "ErrorByShiftDisplay",
    "PrevalenceDisplay",
    "ConfidenceRegionDisplay",
]
