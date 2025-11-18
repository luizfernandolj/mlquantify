.. _counters_module:

.. currentmodule:: mlquantify.adjust_counting


===========================
Counters For Quantification
===========================

To deal with problems of quantification, a straightforward approach is to count the number of items predicted to belong to each class in the unlabeled set. This is the basis of the **Classify and Count** family of methods.


Classify and Count  
==================

The **Classify and Count** method, or :class:`CC` is the simplest baseline.  
It trains a hard classifier :math:`h` on labeled data :math:`L` , applies it to an unlabeled set :math:`U` , and counts how many samples belong to each predicted class.


**Example**

.. code-block:: python

   from mlquantify.adjust_counting import CC
   from sklearn.linear_model import LogisticRegression
   import numpy as np

   X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
   q = CC(learner=LogisticRegression())
   q.fit(X, y)
   q.predict(X)
   # -> {0: 0.47, 1: 0.53}


.. alert::
   :class:`CC` is fast and simple, but when class proportions in the test set differ from the training set, its estimates can become biased or inaccurate.



Probabilistic Classify and Count  
================================

The **Probabilistic Classify and Count** or :class:`PCC` variant uses the *predicted probabilities* from a soft classifier instead of hard labels.  
This makes it less sensitive to uncertain predictions.

[Plot Idea: A plot comparing probabilities per sample and their averaged mean per class]

**Example**

.. code-block:: python

   from mlquantify.adjust_counting import PCC
   from sklearn.linear_model import LogisticRegression
   import numpy as np

   X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
   q = PCC(learner=LogisticRegression())
   q.fit(X, y)
   q.predict(X)
   # -> {0: 0.45, 1: 0.55}

CC and PCC both often underestimate or overestimate the true prevalence when there is distribution shift (also known as "dataset shift").