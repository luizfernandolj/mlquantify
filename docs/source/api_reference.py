
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
                    "get_config",
                    "set_config",
                    "config_context",
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
    "mlquantify.base_aggregative": {
        "short_summary": "Aggregative quantifiers base classes.",
        "description": None,
        "sections": [
            {
                "title": None,
                "autosummary": [
                    "AggregationMixin",
                    "SoftPredictionMixin",
                    "CrispPredictionMixin"
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
                    "binary_quantifier",
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
    "mlquantify.counting": {
        "short_summary": "Counting methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Counting Methods",
                "autosummary": [
                    "CC",
                    "PCC",
                    "ACC",
                    "ThresholdAdjustment",
                    "TAC",
                    "TX",
                    "TMAX",
                    "T50",
                    "MS",
                    "MS2",
                    "FM",
                    "GACC",
                    "GPACC",
                    "evaluate_thresholds",
                    "compute_tpr",
                    "compute_fpr",
                    "compute_table",
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
                    "CDE",
                    "EMQ",
                    "MLPE",
                ],
            }
        ],
    },
    "mlquantify.matching": {
        "short_summary": "Distribution matching methods for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Matching Methods",
                "autosummary": [
                    "BaseMatchingQuantifier",
                    "MatchingHistogramQuantifier",
                    "DyS",
                    "HDy",
                    "HDx",
                    "SORD",
                    "MatchingKernelQuantifier",
                    "MMD_RKHS",
                    "KDEyQuantifier",
                    "KDEyML",
                    "KDEyHD",
                    "KDEyCS",
                    "GKDEyML",
                    "GHDx",
                    "GHDy",
                    "SMM",
                    "EDy",
                    "EDx",
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
                    "PWK",
                ],
            }
        ],
    },
    "mlquantify.compose": {
        "short_summary": "Composable quantification methods.",
        "description": None,
        "sections": [
            {
                "title": "Composable Methods",
                "autosummary": [
                    "BaseComposeQuantifier",
                    "LinearComposeQuantifier",
                    "LikelihoodComposeQuantifier",
                    "ComposeQuantifier",
                ],
            }
        ],
    },
    "mlquantify.losses": {
        "short_summary": "Loss functions used by quantifiers.",
        "description": None,
        "sections": [
            {
                "title": "Loss Functions",
                "autosummary": [
                    "BaseLoss",
                    "DistanceLoss",
                    "LeastSquaresLoss",
                    "HellingerSurrogateLoss",
                    "EnergyLoss",
                    "NegativeLogLikelihoodLoss",
                    "MixtureNegativeLogLikelihoodLoss",
                    "RegularizedMixtureNLLLoss",
                    "normalize_distribution",
                    "get_loss",
                ],
            }
        ],
    },
    "mlquantify.representations": {
        "short_summary": "Representation strategies for quantification.",
        "description": None,
        "sections": [
            {
                "title": "Representations",
                "autosummary": [
                    "BaseRepresentation",
                    "HistogramRepresentation",
                    "KDERepresentation",
                    "DistanceRepresentation",
                    "KernelMeanRepresentation",
                    "PredictionRepresentation",
                    "HardPredictionRepresentation",
                    "SoftPredictionRepresentation",
                ],
            }
        ],
    },
    "mlquantify.solvers": {
        "short_summary": "Optimization helpers for prevalence estimation.",
        "description": None,
        "sections": [
            {
                "title": "Solvers",
                "autosummary": [
                    "solve_binary",
                    "ternary_search",
                    "solve_simplex",
                    "minimize_prevalence",
                    "minimize_prevalence_blocks",
                ],
            }
        ],
    },
    "mlquantify.calibration": {
        "short_summary": "Calibration utilities.",
        "description": None,
        "sections": [
            {
                "title": "Calibration",
                "autosummary": [
                    "Calibrator",
                    "ClassifierCalibrator",
                    "QuantifierCalibrator",
                ],
            }
        ],
    },
    "mlquantify.neural": {
        "short_summary": "Neural quantification methods.",
        "description": None,
        "sections": [
            {
                "title": "Neural Methods",
                "autosummary": [
                    "QuaNet",
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
                    "apply_protocol",
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
                    "normalize_prevalence",
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
