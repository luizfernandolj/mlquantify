try:
	import torch as _torch
except ImportError:  # pragma: no cover - runtime-dependent
	_torch = None

if _torch is not None:
	from ._classes import QuaNet

	__all__ = ["QuaNet"]
else:
	__all__ = []