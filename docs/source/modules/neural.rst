.. _neural_quantifiers:

.. currentmodule:: mlquantify.neural

==================
Neural Quantifiers
==================

Neural quantifiers learn a direct mapping from a bag of instances to a
prevalence vector. The main implementation is QuaNet, which uses an LSTM over
instance embeddings and incorporates auxiliary quantification statistics.

Available methods
=================

- :class:`QuaNet`

Example
=======

.. code-block:: python

   # Requires PyTorch and an estimator with a transform() method
   from mlquantify.neural import QuaNet

   q = QuaNet(estimator=my_embedding_classifier, device="cpu")
   q.fit(X_train, y_train)
   prevalence = q.predict(X_test)
