try:
	import torch as _torch
except ImportError:  # pragma: no cover - runtime-dependent
	_torch = None

if _torch is not None:
	from ._classes import QuaNet, HistNetQ, GMNet
	from ._wrapper import TorchClassifierWrapper
	from ._bag import PrevalenceBagMixin, HistNetQBags, GMNetBags

	__all__ = [
		"QuaNet",
		"HistNetQ",
		"GMNet",
		"TorchClassifierWrapper",
		"PrevalenceBagMixin",
		"HistNetQBags",
		"GMNetBags",
	]
else:
	__all__ = []
