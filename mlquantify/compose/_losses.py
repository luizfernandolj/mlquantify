from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable


def _least_squares(prevalences, q, M, N=None):
    import jax.numpy as jnp

    diff = q - jnp.dot(M, prevalences)
    return jnp.dot(diff, diff)


def _nonzero_features(M):
    import jax.numpy as jnp

    return jnp.any(M != 0, axis=1)


class AbstractLoss(ABC):
    """Base class for composable QUnfold-style losses."""

    @abstractmethod
    def instantiate(self, q, M, N=None):
        """Create a JAX-compatible loss function over class prevalences."""


@dataclass
class FunctionLoss(AbstractLoss):
    """Create a composable loss from a JAX-compatible function."""

    loss_function: Callable

    def instantiate(self, q, M, N=None):
        import jax.numpy as jnp

        M = jnp.asarray(M)
        q = jnp.asarray(q)
        nonzero = _nonzero_features(M)
        M = M[nonzero, :]
        q = q[nonzero]

        return lambda prevalences: self.loss_function(prevalences, q, M, N)


class LeastSquaresLoss(FunctionLoss):
    """Least-squares loss used by ACC and PACC."""

    def __init__(self):
        super().__init__(_least_squares)
