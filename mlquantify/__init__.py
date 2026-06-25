"mlquantify, a Python package for quantification"

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("mlquantify")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

from . import neighbors
from . import likelihood
from . import matching
from . import meta
from . import counting
from . import datasets
from . import model_selection
from . import base_aggregative
from . import base
from . import calibration
from . import confidence
from . import multiclass
from . import losses
try:
    from . import neural
except NameError:
    pass

from ._config import get_config, set_config, config_context

__all__ = [
    "__version__",
    "neighbors",
    "likelihood",
    "matching",
    "meta",
    "counting",
    "datasets",
    "model_selection",
    "base_aggregative",
    "base",
    "calibration",
    "confidence",
    "multiclass",
    "losses",
    "neural",
    "get_config",
    "set_config",
    "config_context",
]
