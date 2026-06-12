"""Tests for prevalence utilities (normalize_prevalence)."""
import warnings

import numpy as np
import pytest

from mlquantify.utils import normalize_prevalence


def test_normalize_prevalence_array():
    assert normalize_prevalence([2.0, 3.0, 5.0], classes=[0, 1, 2]) == pytest.approx(
        {0: 0.2, 1: 0.3, 2: 0.5})


def test_normalize_prevalence_dict():
    assert normalize_prevalence({0: 0.1, 1: 0.1, 2: 0.3}, classes=[0, 1, 2]) == pytest.approx(
        {0: 0.2, 1: 0.2, 2: 0.6})


def test_normalize_prevalence_emits_no_warning():
    """The old implementation raised numpy's 'where without out' UserWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        warnings.simplefilter("error", RuntimeWarning)
        normalize_prevalence([0.4, 0.6], classes=[0, 1])
        normalize_prevalence(np.array([1.0, 2.0, 1.0]), classes=[0, 1, 2])
        normalize_prevalence({0: 1.0, 1: 1.0}, classes=[0, 1])


@pytest.mark.parametrize("prev", [[0.0, 0.0, 0.0], {0: 0.0, 1: 0.0, 2: 0.0}])
def test_normalize_prevalence_zero_sum_is_uniform(prev):
    """A non-positive sum must yield a valid (uniform) vector, not garbage."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = normalize_prevalence(prev, classes=[0, 1, 2])
    vals = np.array([result[c] for c in (0, 1, 2)], dtype=float)
    assert np.all(np.isfinite(vals))
    assert vals.sum() == pytest.approx(1.0)
    assert vals == pytest.approx([1 / 3, 1 / 3, 1 / 3])
