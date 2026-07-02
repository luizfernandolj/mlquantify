"""Collection controls for ``pytest --doctest-modules mlquantify/``.

Modules that require PyTorch at import time cannot be collected when torch
is not installed (the package ``__init__`` files already guard the public
imports); skip them here so the doctest run works on torch-less
environments, mirroring the CI test matrix.
"""

collect_ignore_glob = []

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - runtime-dependent
    collect_ignore_glob += [
        "readme/_readme2.py",
        "representations/_torch_*.py",
    ]
