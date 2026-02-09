
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
        "sections": [
            {
                "title": None,
                "autosummary": [
                ],
            }
        ],
    },
    "mlquantify.base": {
        "short_summary": "Base classes functions for quantifiers.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "BaseQuantifier",
                    "MetaquantifierMixin",
                    "ProtocolMixin",
                ],
            }
        ],
    },
    "mlquantify.config": {
        "short_summary": "Configuration for mlquantify.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "get_config",
                    "set_config",
                    "config_context",
                ],
            }
        ],
    },
    "mlquantify.base_aggregative": {
        "short_summary": "Aggregative quantifiers base classes.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "AggregationMixin",
                    "SoftLearnerQMixin",
                    "CrispLearnerQMixin"
                ],
            }
        ],
    },
    "mlquantify.multiclass": {
        "short_summary": "Multiclass definitions and utilities.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "define_binary",
                    "BinaryQuantifier"
                ],
            }
        ],
    },
    "mlquantify.confidence": {
        "short_summary": "Confidence Regions for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Confidence Regions",
                "autosummary": [
                    "BaseConfidenceRegion",
                    "ConfidenceInterval",
                    "ConfidenceEllipseSimplex",
                    "ConfidenceEllipseCLR",
                    "construct_confidence_region"
                ],
            }
        ],
    },
    "mlquantify.adjust_counting": {
        "short_summary": "Adjusted Counting methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Adjust Counting Methods",
                "autosummary": [
                    "CC",
                    "PCC",
                    "FM",
                    "AC",
                    "PAC",
                    "TAC",
                    "TX",
                    "TMAX",
                    "T50",
                    "MS",
                    "MS2",
                    "CDE",
                    "evaluate_thresholds",
                    "compute_tpr",
                    "compute_fpr",
                    "compute_table"
                ],
            }
        ],
    },
    "mlquantify.likelihood": {
        "short_summary": "Likelihood methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Likelihood Methods",
                "autosummary": [
                    "EMQ"
                ],
            }
        ],
    },
    "mlquantify.mixture": {
        "short_summary": "Mixture Models for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Mixture Models",
                "autosummary": [
                    "HDy",
                    "DyS",
                    "SMM",
                    "SORD",
                    "HDx",
                    "MMD_RKHS"
                ],
            }
        ],
    },
    "mlquantify.neighbors": {
        "short_summary": "Neighbor-based methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Neighbor-based Methods",
                "autosummary": [
                    "KDEyCS",
                    "KDEyHD",
                    "KDEyML",
                    "PWK"
                ],
            }
        ],
    },
    "mlquantify.meta": {
        "short_summary": "Meta methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Meta Methods",
                "autosummary": [
                    "EnsembleQ", 
                    "QuaDapt", 
                    "AggregativeBootstrap"
                ],
            }
        ],
    },
    "mlquantify.metrics": {
        "short_summary": "Metrics for quantification.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "AE",
                    "SE",
                    "MAE",
                    "MSE",
                    "KLD",
                    "RAE",
                    "NAE",
                    "NRAE",
                    "NKLD",
                    "NMD",
                    "RNOD",
                    "VSE",
                    "CvM_L1",
                ],
            }
        ],
    },
    "mlquantify.model_selection": {
        "short_summary": "Model selection methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "GridSearchQ",
                    "BaseProtocol",
                    "APP",
                    "NPP",
                    "UPP",
                    "PPP",
                ],
            }
        ],
    },
    "mlquantify.utils": {
        "short_summary": "Utility functions for quantification.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "get_prev_from_labels",
                    "load_quantifier",
                    "make_prevs",
                    "apply_cross_validation",
                    "simplex_uniform_kraemer",
                    "simplex_grid_sampling",
                    "simplex_uniform_sampling",
                    "get_indexes_with_prevalence"
                ],
            }
        ]
    },
}


