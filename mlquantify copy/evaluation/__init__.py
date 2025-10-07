from . import measures


MEASURES = {
    "ae": measures.absolute_error,
    "mae": measures.mean_absolute_error,
    "nae": measures.normalized_absolute_error,
    "kld": measures.kullback_leibler_divergence,
    "nkld": measures.normalized_kullback_leibler_divergence,
    "nrae": measures.normalized_relative_absolute_error,
    "rae": measures.relative_absolute_error,
    "se": measures.squared_error,
    "mse": measures.mean_squared_error
}