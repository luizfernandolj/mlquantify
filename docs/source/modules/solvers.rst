.. _solvers:

.. currentmodule:: mlquantify.solvers

=======
Solvers
=======

Solver utilities minimize objective functions under prevalence constraints.
They are used by compose and matching quantifiers to enforce simplex
constraints.

Available solvers
=================

- :func:`solve_binary`
- :func:`ternary_search`
- :func:`solve_simplex`
- :func:`minimize_prevalence`
- :func:`minimize_prevalence_blocks`

Example
=======

.. code-block:: python

   import numpy as np
   from mlquantify.solvers import minimize_prevalence

   def objective(p):
       target = np.array([0.4, 0.6])
       return np.sum((p - target) ** 2)

   prevalences, value = minimize_prevalence(
       objective=objective,
       n_classes=2,
       solver="slsqp",
       random_state=0,
   )
