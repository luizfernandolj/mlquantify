from .ae import absolute_error
from .kld import kullback_leibler_divergence
from .nkld import normalized_kullback_leibler_divergence
from .rae import relative_absolute_error
from .nae import normalized_absolute_error
from .bias import bias
from .nrae import normalized_relative_absolute_error
from .se import squared_error
from .mse import mean_squared_error



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