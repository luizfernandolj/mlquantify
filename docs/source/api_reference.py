
def _get_submodule(module_name, submodule_name):
    """Get the submodule docstring and automatically add the hook.

    `module_name` is e.g. `sklearn.feature_extraction`, and `submodule_name` is e.g.
    `image`, so we get the docstring and hook for `sklearn.feature_extraction.image`
    submodule. `module_name` is used to reset the current module because autosummary
    automatically changes the current module.
    """
    lines = [
        f".. automodule:: {module_name}.{submodule_name}",
        f".. currentmodule:: {module_name}",
    ]
    return "\n\n".join(lines)

API_REFERENCE = {
    "mlquantify": {
        "short_summary": "A library for quantifying machine learning models.",
        "description": None,
        "section": [
            {
                "title": None,
                "autosummary": [
                    "set_arguments",
                ],
            }
        ],
    },
    "mlquantify.base": {
        "short_summary": "Base classes functions for quantifiers.",
        "description": None,
        "section": [
            {
                "title": None,
                "autosummary": [
                    "Quantifier",
                    "AggregativeQuantifier",
                    "NonAggregativeQuantifier",
                ],
            }
        ],
    },
    "mlquantify.methods.aggregative": {
        "short_summary": "Aggregative quantification methods.",
        "description": None,
        "section": [
            {
                "title": "Aggregative methods",
                "autosummary": [
                    "CC", "EMQ", "FM", "GAC", "GPAC", "PCC", "PWK",
                    "ACC", "MAX", "MS", "MS2", "PACC", "T50", "X_method",
                    "DyS", "DySsyn", "HDy", "SMM", "SORD"
                ],
            }
        ],
    },
    "mlquantify.methods.non_aggregative": {
        "short_summary": "Non-aggregative quantification methods.",
        "description": None,
        "section": [
            {
                "title": "Non-aggregative methods",
                "autosummary": [
                    "HDx"
                ],
            }
        ],
    },
    "mlquantify.methods.meta": {
        "short_summary": "Meta quantification methods.",
        "description": None,
        "section": [
            {
                "title": "Meta methods",
                "autosummary": [
                    "Ensemble"
                ],
            }
        ],
    },
    "mlquantify.classification.methods": {
        "short_summary": "Classification methods for quantification algorithms.",
        "description": None,
        "section": [
            {
                "title": None,
                "autosummary": [
                    "PWKCLF"
                ],
            }
        ],
    },
    "mlquantify.evaluation.measures": {
        "short_summary": "Evaluation metrics for quantification.",
        "description": None,
        "section": [
            {
                "title": None,
                "autosummary": [
                    "process_inputs",
                    "absolute_error",
                    "mean_absolute_error",
                    "kullback_leibler_divergence",
                    "squared_error",
                    "mean_squared_error",
                    "normalized_absolute_error",
                    "normalized_kullback_leibler_divergence",
                    "relative_absolute_error",
                    "normalized_relative_absolute"
                ],
            }
        ],
    },
    "mlquantify.evaluation.protocol": {
        "short_summary": "Evaluation metrics for quantification.",
        "description": None,
        "section": [
            {
                "title": None,
                "autosummary": [
                    "Protocol", "APP", "NPP"
                ],
            }
        ],
    },
    "mlquantify.utils.general": {
        "short_summary": "Utility functions for quantification.",
        "description": None,
        "section": [
            {
                "title": None,
                "autosummary": [
                    "convert_columns_to_arrays",
                    "generate_artificial_indexes",
                    "generate_artificial_prevalences",
                    "get_real_prev",
                    "load_quantifier",
                    "make_prevs",
                    "normalize_prevalence",
                    "parallel",
                    "round_protocol_df",
                    "get_measure",
                    "get_method"
                ],
            }
        ],
    },
    "mlquantify.utils.method": {
        "short_summary": "Utility functions for quantification.",
        "description": None,
        "section": [
            {
                "title": None,
                "autosummary": [
                    "sqEuclidean",
                    "probsymm",
                    "topsoe",
                    "hellinger",
                    "get_scores",
                    "getHist",
                    "MoSS",
                    "ternary_search",
                    "compute_table",
                    "compute_tpr",
                    "compute_fpr",
                    "adjust_threshold"
                ],
            }
        ],
    },
    "mlquantify.model_selection": {
        "short_summary": "Model selection methods for quantification.",
        "description": None,
        "section": [
            {
                "title": None,
                "autosummary": [
                    "GridSearchQ"
                ],
            }
        ],
    }
}
