# losses/_base.py

from abc import ABC, abstractmethod


class BaseLoss(ABC):
    """Base class for optimization losses."""

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Compute the loss."""
        ...