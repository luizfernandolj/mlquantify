from ._classes import ReadMe

__all__ = ["ReadMe"]

try:
    import torch as _torch
except ImportError:  # pragma: no cover - runtime-dependent
    _torch = None

if _torch is not None:
    from ._readme2 import ReadMe2

    __all__.append("ReadMe2")
