from ._base import BaseComposeQuantifier
from ._linear import LinearComposeQuantifier
from ._likelihood import LikelihoodComposeQuantifier

ComposeQuantifier = LinearComposeQuantifier

__all__ = [
    "BaseComposeQuantifier",
    "LinearComposeQuantifier",
    "LikelihoodComposeQuantifier",
    "ComposeQuantifier",
]