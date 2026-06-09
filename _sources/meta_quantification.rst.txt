.. _meta_quantification:

Meta-Quantification
--------------------

Meta-quantification methods wrap an existing base quantifier and add a
higher-level strategy — ensembling, adaptive score correction, or bootstrap
confidence estimation — to improve accuracy or reliability.

They do not replace the base quantifier; they *augment* it. This means you
can take any method from the aggregative or distribution-matching families
and wrap it with a meta-quantifier to gain extra robustness or uncertainty
estimates with minimal code changes.

.. code-block:: python

   # Take any quantifier...
   from mlquantify.matching import DyS
   from sklearn.linear_model import LogisticRegression
   base = DyS(LogisticRegression())

   # ...and wrap it with a meta-quantifier
   from mlquantify.meta import EnsembleQ
   q = EnsembleQ(base, size=30, n_jobs=-1)
   q.fit(X_train, y_train)
   print(q.predict(X_test))

See :ref:`quantification_foundations` for an overview of when meta-methods
provide the most benefit.

.. toctree::
   :maxdepth: 2

   Ensembles and Adaptation <modules/ensemble.rst>
   modules/bootstrap.rst
   modules/scores_adaptation.rst