.. _classify_count:

Classify Count
==============

.. currentmodule:: mlquantify.methods.aggregative.CC

The most basic quantification method is the Classify and Count (CC) method, which is a simple approach that counts the number of instances of each class in the test set. The CC method is based on the assumption that the class distribution in the test set is similar to that in the training set. The CC method can be used as a baseline for comparison with other quantification methods.
The CC method is implemented in the :class:`~mlquantify.methods.aggregative.CC` class.