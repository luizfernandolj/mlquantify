<h1 align="center">MLQuantify</h1>
<h4 align="center">A Python Package for Quantification</h4>

___

 **mlquantify** is a Python library for quantification, also known as supervised prevalence estimation, designed to estimate the distribution of classes within datasets. It offers a range of tools for various quantification methods, model selection tailored for quantification tasks, evaluation metrics, and protocols to assess quantification performance. Additionally, mlquantify includes popular datasets and visualization tools to help analyze and interpret results.

___

## Latest Release

- **Version 0.0.11.6**: Inicial beta version. For a detailed list of changes, check the [changelog](#).
- In case you need any help, refer to the [wiki](https://github.com/luizfernandolj/mlquantify/wiki).
- Explore the [API documentation](#) for detailed developer information.
- See also the library in the pypi site in [pypi mlquantify](https://pypi.org/project/mlquantify/)

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
| **21 Quantification Methods** | Methods for quantification, such as classify & Count Correct methods, Threshold Optimization, Mixture Models and more.|
| **Dynamic class management** | All methods are dynamic, and handles multiclass and binary problems, in case of binary it makes One-Vs-All (OVA) automatically. |
| **Model Selection** | Criteria and processes used to select the best model, such as grid-search for the case of quantification|
| **Evaluation Metrics** | Specific metrics used to evaluate quantification performance, (e.g., AE, BIAS, NAE, SE, KLD, etc.). |
| **Evaluation Protocols** | Evaluation protocols used, based on sampling generation (e.g., APP, NPP, etc.).. |
| **Plotting Results** | Tools and techniques used to visualize results, such as the protocol results.|
| **Comprehensive Documentation** | Complete documentation of the project, including code, data, and results. |

___

## Quick example:

This code first loads the breast cancer dataset from _sklearn_, which is then split into training and testing sets. It uses the _Expectation Maximisation Quantifier (EMQ)_ with a RandomForest classifier to predict class prevalence. After training the model, it evaluates performance by calculating and printing the absolute error and bias between the real and predicted prevalences.

```python
import mlquantify as mq
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Loading dataset from sklearn
features, target = load_breast_cancer(return_X_y=True)

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3)

#Create the model, here it is the Expectation Maximisation Quantifier (EMQ) with a classifier
model = mq.methods.EMQ(RandomForestClassifier())
model.fit(X_train, y_train)

#Predict the class prevalence for X_test
pred_prevalence = model.predict(X_test)
real_prevalence = mq.utils.get_real_prev(y_test)

#Get the error for the prediction
ae = mq.evaluation.absolute_error(real_prevalence, pred_prevalence)
bias = mq.evaluation.bias(real_prevalence, pred_prevalence)

print(f"Mean Squared Error (MSE) -> {ae:.4f}")
print(f"Bias -> {bias}")
```

___

## Requirements

- Scikit-learn
- pandas
- numpy
- joblib
- tqdm
- matplotlib
- xlrd

___

## Documentation

##### API is avaliable [here](#)

- [Methods](https://github.com/luizfernandolj/mlquantify/wiki/Methods)
- [Model Selection](https://github.com/luizfernandolj/mlquantify/wiki/Model-Selection)
- [Evaluation](https://github.com/luizfernandolj/mlquantify/wiki/Evaluation)
- [Plotting](https://github.com/luizfernandolj/mlquantify/wiki/Plotting)
- [Utilities](https://github.com/luizfernandolj/mlquantify/wiki/Utilities)


___
