.. _using_aggregative:

.. currentmodule:: mlquantify.adjust_counting

========================================
Using Aggregative Quantification Methods
========================================

This guide explains how to use the aggregative quantification methods in mlquantify.


========================================
General Concept
========================================

Aggregative quantification refers to methods that estimate the prevalence of classes in a dataset by aggregating the predictions made on individual data items. Except for special-purpose learning methods, all aggregative methods share a common structure involving three main steps:

1. **fit**: the model is trained on labeled data.
2. **predict**: the trained model generates predictions for each individual item in the unlabeled dataset. These predictions can be either hard labels or soft probabilities.
3. **aggregate**: this step uses the results from the prediction phase to estimate the class prevalence distribution in the dataset. The `aggregate` function takes as input the necessary information from the classification results to perform this aggregation.

This clear separation into fit, predict, and aggregate methods allows modular implementation and reuse of standard classifiers as a backbone.

The main aggregative quantification methods will be defined next, with details about their specific approaches and differences.

.. warning::

    All methods in this library can work with different types of prediction inputs.
    For example, :class:`CC` accepts either hard class labels or soft class probabilities, while :class:`PCC` requires soft probabilities only.
    In contrast, :class:`AC` uses soft probabilities together with the class prior probabilities and the true training labels.


Examples
--------

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from mlquantify.counters import AC

   # Create a classifier
   clf = LogisticRegression()

   ac = AC()

   # used when you just have the sample to predict
   ac.fit(X_train, y_train)
   prevalence = ac.predict(X_test)

   # used when you already have the sample predictions (usually from cross-validation for training predictions)
   prevalence = ac.aggregate(posteriors, train_posteriors, y_train)






.. toctree::
   :maxdepth: 2

   modules/counters.rst
   modules/adjust_counting.rst
   modules/likelihood.rst
   modules/mixture_models.rst
   modules/neighbors.rst
   modules/density.rst
