"""Root conftest: make docstring doctests deterministic and config-consistent.

Doctests scattered through the package illustrate prevalence outputs as dicts
(e.g. ``{0: 0.47, 1: 0.53}``) and use randomly generated toy data. This fixture
seeds NumPy and sets the dict return format before each doctest runs, so the
examples are reproducible regardless of the global default (which is ``array``).
"""

import numpy as np
import pytest

from mlquantify import set_config


@pytest.fixture(autouse=True)
def _doctest_determinism(request):
    if request.node.__class__.__name__ == "DoctestItem":
        np.random.seed(0)
        set_config(prevalence_return_type="dict")
        try:
            yield
        finally:
            set_config(prevalence_return_type="array")
    else:
        yield


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    # Make ``np`` available to examples that reference it without importing.
    doctest_namespace["np"] = np
    doctest_namespace["numpy"] = np
