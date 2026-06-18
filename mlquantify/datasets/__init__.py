"""Synthetic and helper datasets for quantification.

Mirrors the spirit of :mod:`sklearn.datasets`, but the generators produce
*bags* (samples with controlled class prevalence) suitable for evaluating
quantifiers under distribution shift.
"""

from ._samples_generator import make_quantification

__all__ = ["make_quantification"]
