.. _compose_quantifiers:

.. currentmodule:: mlquantify.compose

====================
Composable Quantifiers
====================

Compose quantifiers build prevalence estimators by combining a representation,
a loss function, and a solver. This makes it easy to swap components when you
want to experiment with alternative representations or objectives.

Key classes
===========

- :class:`BaseComposeQuantifier`
- :class:`LinearComposeQuantifier` (alias :class:`ComposeQuantifier`)
- :class:`LikelihoodComposeQuantifier`

Linear compose example
======================

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from mlquantify.compose import LinearComposeQuantifier
   from mlquantify.representations import HistogramRepresentation

   representation = HistogramRepresentation(bins=(10,))
   q = LinearComposeQuantifier(
       representation=representation,
       estimator=LogisticRegression(),
       loss="least_squares",
   )

   q.fit(X_train, y_train)
   prevalence = q.predict(X_test)

Likelihood compose example
==========================

.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from mlquantify.compose import LikelihoodComposeQuantifier

   q = LikelihoodComposeQuantifier(
       estimator=LogisticRegression(),
       tau_0=0.1,
       tau_1=0.0,
   )

   q.fit(X_train, y_train)
   prevalence = q.predict(X_test)
