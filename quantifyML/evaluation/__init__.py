from .measures import *
from .protocol import *

MEASURES = {
    "ae": absolute_error,
    "nae": normalized_absolute_error,
    "kld": kullback_leibler_divergence,
    "nkld": normalized_kullback_leibler_divergence,
    "nrae": normalized_relative_absolute_error,
    "rae": relative_absolute_error,
    "se": squared_error,
    "mse": mean_squared_error
}


def get_measure(measure:str):
    return MEASURES.get(measure)