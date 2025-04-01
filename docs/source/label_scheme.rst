.. _label_scheme:

Label Scheme
------------

The library is designed to work with binary and multiclass problems, with different approaches for each, determined dinamically based on the number of classes in the train set. Below is a summary of the label scheme used in the library:


.. _binary_problems:

Binary problems
==================

All methods in the library work with binary problems. A binary problem is a classification problem where the target variable has only two possible values (e.g., 0 and 1, or "yes" and "no"). In this context, the library provides tools for quantifying the class distribution of the target variable in a given dataset. The usage of the library for binary problems is similar to that for multiclass problems, with the main difference being the number of classes involved and the funcionality of some quantifiers.

.. _multiclass_problems:

Multiclass Problems
===================

In multiclass problems, most methods that only work with binary problems can still be used, applying them to each class separately, using the one-vs-all approach. This means that for each class, the method is applied as if it were a binary problem, treating the selected class as the positive class and all other classes as the negative class. The results are normalized to obtain the final class distribution.