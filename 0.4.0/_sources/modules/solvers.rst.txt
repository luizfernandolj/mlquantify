.. _solvers:

.. currentmodule:: mlquantify.solvers

=======
Solvers
=======

Solvers are the optimisation backends shared by the quantifier families. They
turn a distribution-matching or likelihood objective into a prevalence vector,
either over the binary interval :math:`[0, 1]` or over the full probability
simplex :math:`\Delta^{k-1} = \{p : p_c \ge 0,\; \sum_c p_c = 1\}`. Compose and
matching quantifiers delegate their final optimisation step to one of these
functions, so the same solver code is reused across many methods.

Role and mechanism
==================

Every optimisation-based quantifier reduces prevalence estimation to

.. math::

   \hat{p} = \arg\min_{p \in \Delta^{k-1}} \mathcal{L}(p),

where :math:`\mathcal{L}` is a loss comparing the test representation with the
prevalence-mixed training representations. The solvers differ only in *how* they
search that feasible set.

.. list-table::
   :widths: 28 20 52
   :header-rows: 1

   * - Solver
     - Search space
     - Mechanism / when used
   * - :func:`solve_binary`
     - :math:`[0, 1]`
     - Scalar search over the positive prevalence (grid, ternary, or bounded).
       Backbone of the binary matching methods.
   * - :func:`ternary_search`
     - interval
     - Derivative-free trisection for a unimodal objective; the engine behind
       ``solve_binary(solver="ternary")``.
   * - :func:`solve_simplex`
     - simplex
     - SLSQP with the equality constraint :math:`\sum_c p_c = 1` and box bounds;
       the multiclass workhorse.
   * - :func:`minimize_prevalence`
     - both
     - Dispatcher: routes binary problems to :func:`solve_binary` and multiclass
       problems to :func:`solve_simplex`.
   * - :func:`minimize_prevalence_blocks`
     - simplex (per block)
     - Minimises one estimate per representation block (e.g. a histogram bin
       group) and aggregates them by the median; the recipe behind HDy/HDx.

Choosing a solver
=================

- Leave ``solver="auto"`` for most cases: it selects a bounded scalar search for
  binary problems and SLSQP for multiclass problems.
- Use ``"grid"`` when the objective may be multi-modal and you want a global
  search; use ``"ternary"`` when it is unimodal and you want speed.
- :func:`minimize_prevalence` is the entry point a quantifier should call, so the
  same code runs unchanged in both the binary and multiclass settings.

Used by
=======

Binary matching (:class:`~mlquantify.matching.DyS`,
:class:`~mlquantify.matching.HDy`, :class:`~mlquantify.matching.SORD`) goes
through :func:`solve_binary`; the constrained-regression and density methods
(:class:`~mlquantify.counting.GACC`, :class:`~mlquantify.matching.MMD_RKHS`,
:class:`~mlquantify.matching.KDEyML`, :class:`~mlquantify.matching.EDy`) go
through :func:`solve_simplex`; the histogram sweep
(:class:`~mlquantify.matching.HDy` / :class:`~mlquantify.matching.HDx`) uses
:func:`minimize_prevalence_blocks`.

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

.. seealso::

   :ref:`distribution_matching` and :ref:`counters_module` for the quantifiers
   that delegate their optimisation to these solvers.
