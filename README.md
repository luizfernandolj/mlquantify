![PyPI - Version](https://img.shields.io/pypi/v/mlquantify)
[![docs](https://img.shields.io/badge/docs-sphinx-blue)](https://luizfernandolj.github.io/mlquantify/)


<a href="https://luizfernandolj.github.io/mlquantify/"><img src="assets/logo_mlquantify-white.svg" alt="mlquantify logo"></a>
<h4 align="center">A Python Package for Quantification</h4>

___

 **mlquantify** is a Python library for quantification, also known as supervised prevalence estimation, designed to estimate the distribution of classes within datasets. It offers a range of tools for various quantification methods, model selection tailored for quantification tasks, evaluation metrics, and protocols to assess quantification performance. Additionally, mlquantify includes calibration tools, confidence region estimation, pluggable solvers and representations, and visualization utilities to help analyze and interpret results.

 Website: https://luizfernandolj.github.io/mlquantify/

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
| **Comprehensive Documentation** | Full API reference and user guide covering all modules and methods. |

___

## Quick example:

This code first loads the breast cancer dataset from _sklearn_, which is then split into training and testing sets. It uses the _Expectation Maximisation Quantifier (EMQ)_ with a RandomForest classifier to predict class prevalence. After training the model, it evaluates performance by calculating and printing the absolute error and bias between the real and predicted prevalences.

```python
from mlquantify.likelihood import EMQ
from mlquantify.metrics import MAE, NRAE
from mlquantify.utils import get_prev_from_labels

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Loading dataset from sklearn
features, target = load_breast_cancer(return_X_y=True)

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

#Create the model, here it is the Expectation Maximisation Quantifier (EMQ) with a classifier
model = EMQ(RandomForestClassifier())
model.fit(X_train, y_train)

#Predict the class prevalence for X_test
pred_prevalence = model.predict(X_test)
real_prevalence = get_prev_from_labels(y_test)

#Get the error for the prediction
mae = MAE(real_prevalence, pred_prevalence)
nrae = NRAE(real_prevalence, pred_prevalence)

print(f"Mean Absolute Error -> {mae}")
print(f"Normalized Relative Absolute Error -> {nrae}")
```

- In case you need any help, refer to the [User Guide](https://luizfernandolj.github.io/mlquantify/user_guide.html).
- Explore the [API documentation](https://luizfernandolj.github.io/mlquantify/api/index.html) for detailed developer information.
- See also the library in the pypi site in [pypi mlquantify](https://pypi.org/project/mlquantify/)

___

## Requirements

- scikit-learn
- numpy
- scipy
- pandas
- joblib
- tqdm
- matplotlib
- xlrd
- abstention