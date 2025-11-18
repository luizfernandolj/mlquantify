.. _aggregative_quantification:

Aggregative Quantification
--------------------------

Aggregative quantification refers to methods that estimate the prevalence of classes in a dataset by aggregating the predictions made on individual data items. Except for special-purpose learning methods, all aggregative methods share a common structure involving three main steps:

1. **fit**: the model is trained on labeled data.
2. **predict**: the trained model generates predictions for each individual item in the unlabeled dataset. These predictions can be either hard labels or soft probabilities.
3. **aggregate**: this step uses the results from the prediction phase to estimate the class prevalence distribution in the dataset. The `aggregate` function takes as input the necessary information from the classification results to perform this aggregation.

This clear separation into fit, predict, and aggregate methods allows modular implementation and reuse of standard classifiers as a backbone.

The main aggregative quantification methods will be defined next, with details about their specific approaches and differences.


.. toctree::
   :maxdepth: 2

   modules/counters.rst
   modules/adjust_counting.rst
   modules/likelihood.rst
   modules/mixture_models.rst
   modules/neighbors.rst
   modules/density.rst

