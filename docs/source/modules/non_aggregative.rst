.. _non_aggregative:

==============================
Non Aggregative Quantification
==============================

.. currentmodule:: mlquantify.methods.non_aggregative

Non-aggregative quantification methods do not rely on the classifier's predicted values to infer the class distribution of the test set. Instead, they estimate the distribution directly, taking advantage of information from the training set and/or the test set.

Currently, the library implements only the :class:`~mlquantify.methods.non_aggregative.HDx` method. This non-aggregative binary quantifier estimates the class distribution by measuring the distance between training and test features, and it is similar to the :class:`~mlquantify.methods.aggregative.HDy` quantifier proposed by the same author (`González-Castro et al. (2013)`_).

.. _González-Castro et al. (2013):
   https://www.sciencedirect.com/science/article/abs/pii/S0020025512004069?casa_token=W6UksOigmp4AAAAA:ap8FK5mtpAzG-s8k2ygfRVgdIBYDGWjEi70ueJ546coP9F-VNaCKE5W_gsAv0bWQiwzt2QoAuLjP