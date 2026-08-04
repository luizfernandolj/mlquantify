[![unit tests](https://img.shields.io/github/actions/workflow/status/luizfernandolj/mlquantify/tests.yml?branch=master&label=unit%20tests)](https://github.com/luizfernandolj/mlquantify/actions/workflows/tests.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/mlquantify)](https://pypi.org/project/mlquantify/)
[![python versions](https://img.shields.io/pypi/pyversions/mlquantify)](https://pypi.org/project/mlquantify/)
[![docs](https://img.shields.io/badge/docs-sphinx-blue)](https://luizfernandolj.github.io/mlquantify/)


<a href="https://coenlab.github.io/mlquantify/stable/"><img src="assets/logo_mlquantify-white.svg" alt="mlquantify logo"></a>
<h4 align="center">A Python Package for Quantification</h4>

___

 Website: https://luizfernandolj.github.io/mlquantify/

 **mlquantify** is a Python library for quantification, also known as supervised prevalence estimation, designed to estimate the distribution of classes within datasets. It offers a range of tools for various quantification methods, model selection tailored for quantification tasks, evaluation metrics, and protocols to assess quantification performance. Additionally, mlquantify includes calibration tools, confidence region estimation, pluggable solvers and representations, and visualization utilities to help analyze and interpret results.

___

## Installation

To install mlquantify, run the following command:

```bash
pip install mlquantify
```

If you only want to update, run the code below:

```bash
pip install --upgrade mlquantify
```

___

## Contents

| Section | Description |
|---|---|
| **33 Quantification Methods** | Counting (CC, PCC, ACC, TAC, TX, TMAX, T50, MS, MS2, FM, GACC, GPACC), Matching (DyS, HDy, HDx, SORD, SMM, MMD_RKHS, KDEyML, KDEyHD, KDEyCS, GHDy, GHDx, GKDEyML, EDy, EDx), Likelihood (EMQ, CDE, MLPE), Neighbors (PWK), Meta (EnsembleQ, AggregativeBootstrap, QuaDapt). |
| **Dynamic class management** | All methods are dynamic, and handle multiclass and binary problems; in the binary case, One-Vs-All (OVA) is applied automatically. |
| **Solvers** | Modular optimization backends: `BinarySolver`, `LeastSquaresSolver`, `SimplexSolver`. |
| **Representations** | Pluggable feature representations: `HistogramRepresentation`, `KDERepresentation`, `DistanceRepresentation`, `KernelMeanRepresentation`, `PredictionRepresentation`. |
| **Losses** | Composable loss functions (distance-based and likelihood-based) shared across quantifier families. |
| **Calibration** | `ClassifierCalibrator` and `QuantifierCalibrator` for post-hoc calibration of classifiers and quantifiers. |
| **Confidence Regions** | `ConfidenceInterval`, `ConfidenceEllipseSimplex`, `ConfidenceEllipseCLR` for uncertainty estimation on prevalence predictions. |
| **Model Selection** | `GridSearchQ` and evaluation protocols (APP, NPP, UPP, PPP) tailored for quantification tasks. |
| **Evaluation Metrics** | Metrics for quantification performance: AE, MAE, NAE, SE, MSE, KLD, RAE, NRAE, NKLD, NMD, RNOD, VSE, CvM_L1. |
| **Visualization** | scikit-learn-style `Display` classes for both single- and multiple-sample results: `DiagonalDisplay`, `BiasDisplay`, `ErrorByShiftDisplay`, `PrevalenceDisplay`, `ConfidenceRegionDisplay`. |
| **Comprehensive Documentation** | Full API reference and user guide covering all modules and methods. |

___

## Quick example:

This snippet builds a synthetic 3-class dataset that combines **prior** and **covariate** shift with `make_quantification`, then evaluates a quantifier across the **Artificial Prevalence Protocol (APP)** with `apply_protocol` — which trains the model on a held-out split and scores it on many test samples drawn at controlled prevalences.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from mlquantify.datasets import make_quantification
from mlquantify.model_selection import apply_protocol
from mlquantify.counting import ACC

# Synthetic 3-class data with a complex (stacked prior + covariate) shift
Xs, ys, _ = make_quantification(
    n_classes=5,
    shift_type=["prior", "covariate"],   # compose two kinds of dataset shift
    covariate_scale=1.0,
    random_state=0,
)
X, y = np.concatenate(Xs), np.concatenate(ys)   # pool the bags into one dataset

# Evaluate ACC under the Artificial Prevalence Protocol. apply_protocol fits the
# quantifier on a train split and scores it on many held-out test samples drawn
# at controlled prevalences (the test instances never overlap with training).
results = apply_protocol(
    ACC(RandomForestClassifier(random_state=0)), X, y,
    protocol="app", batch_size=[100, 500], scoring=["mae", "rae"], random_state=0, verbose=1
)

print(f"APP over {results['n_batches']} test samples")
print(f"  mean MAE -> {results['MAE'].mean():.4f}")
print(f"  mean RAE -> {results['RAE'].mean():.4f}")
```

- In case you need any help, refer to the [User Guide](https://luizfernandolj.github.io/mlquantify/user_guide.html).
- Explore the [API documentation](https://luizfernandolj.github.io/mlquantify/api/index.html) for detailed developer information.
- See also the library in the pypi site in [pypi mlquantify](https://pypi.org/project/mlquantify/)
- Check the [CHANGELOG](https://github.com/luizfernandolj/mlquantify/blob/master/CHANGELOG.md) to see what's currently beign developed!

___

## Requirements

Core dependencies (installed automatically with `pip install mlquantify`):

- scikit-learn
- numpy
- scipy
- pandas
- joblib
- tqdm

Optional extras:

- `pip install mlquantify[viz]` — plotting via `mlquantify.visualization` (adds matplotlib)
- `pip install mlquantify[neural]` — neural quantifiers such as QuaNet (adds PyTorch)
- `pip install mlquantify[all]` — everything above
