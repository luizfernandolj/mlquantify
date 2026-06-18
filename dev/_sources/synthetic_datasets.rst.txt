.. _synthetic_datasets:

.. currentmodule:: mlquantify.datasets

==================
Synthetic Datasets
==================

:func:`make_quantification` is the quantification analogue of
:func:`sklearn.datasets.make_classification`. Where a classification generator
hands you one labelled table, a *quantification* generator must hand you many
**bags** — test samples whose class proportions vary — together with each bag's
**true prevalence**, because that vector (not the individual labels) is what a
quantifier is scored against.

How it works
------------------------------------------------------------

The generator works in two stages:

1. **Build one labelled population.** Internally it calls
   :func:`~sklearn.datasets.make_classification` once to create a pool of
   ``n_samples`` points, whose difficulty you control with ``class_sep``
   (cluster separation), ``flip_y`` (label noise), ``n_features`` and
   ``weights``.
2. **Draw ``n_batches`` bags from that population**, where each bag's class
   balance, feature positions, or labelling are perturbed according to the
   chosen ``shift_type``.

The call returns three aligned objects:

.. code-block:: python

    from mlquantify.datasets import make_quantification

    Xs, ys, prevalences = make_quantification(n_batches=10, random_state=0)
    #   Xs           : list of feature matrices, one per bag
    #   ys           : list of label vectors,    one per bag
    #   prevalences  : (n_batches, n_classes) array of TRUE prevalences

``prevalences[i]`` is the quantification target for bag ``i`` — feed ``Xs[i]`` to
a fitted quantifier and compare its prediction against ``prevalences[i]``.

Controlling the prevalence (prior shift)
------------------------------------------------------------

By default ``shift_type="prior"``: the class prevalence :math:`P(y)` changes from
bag to bag while the class-conditional features :math:`P(x \mid y)` stay fixed.
The ``prevalence`` argument decides *how* each bag's target prevalence is drawn:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``prevalence``
     - Behaviour
   * - ``"uniform"``
     - Prevalences spread uniformly over the probability simplex — the full
       range of shifts (the default).
   * - ``"grid"``
     - A regular grid over the simplex (the Artificial Prevalence Protocol); the
       bag count is then set by ``n_prevalences`` and ``repeats``.
   * - ``"natural"``
     - Bags drawn i.i.d. from the population, so prevalence fluctuates only with
       sampling noise around ``weights``.
   * - ``"dirichlet"``
     - From a Dirichlet centred on ``target_prevalence`` with spread set by
       ``concentration`` — high concentration pins bags to the target, low
       concentration spreads them out.

.. code-block:: python

    # bags concentrated near a 70/30 split
    Xs, ys, prevs = make_quantification(
        n_batches=30, prevalence="dirichlet",
        target_prevalence=[0.7, 0.3], concentration=150, random_state=0)

Shift types
------------------------------------------------------------

Beyond prior shift, ``shift_type`` can request the other two canonical kinds of
dataset shift. Each is realised by a different per-bag operation on the *same*
fixed population:

.. list-table::
   :header-rows: 1
   :widths: 16 38 46

   * - ``shift_type``
     - What changes
     - How a bag is built
   * - ``"prior"``
     - :math:`P(y)` — the class balance
     - Resample the population to a target prevalence; the labels are the
       population's own.
   * - ``"covariate"``
     - :math:`P(x)` — where the features sit
     - **Translate** the bag's features by a random vector, then label them with
       a *fixed* reference boundary. The boundary stays put while the cloud
       slides across it (``covariate_scale`` sets the distance).
   * - ``"concept"``
     - :math:`P(y \mid x)` — the labelling rule
     - Keep the features in place but **rotate** the reference boundary and
       relabel, so the same point can change class (``concept_strength`` sets the
       rotation).

Both covariate and concept shift use a single **reference decision boundary** — a
logistic-regression rule fit on the population. Covariate shift moves the
features across that boundary; concept shift rotates the boundary itself.

The decision boundary
------------------------------------------------------------

Because that boundary is explicit, ``return_boundary=True`` hands it back, so the
exact labelling rule is known rather than hidden:

.. code-block:: python

    Xs, ys, prevs, boundary = make_quantification(
        n_batches=5, shift_type="covariate", return_boundary=True, random_state=0)

``boundary`` is a ``DecisionBoundary`` namedtuple with per-bag ``coef`` and
``intercept`` arrays. For a **binary** problem ``coef`` has shape
``(n_bags, n_features)`` and ``intercept`` ``(n_bags,)`` — the hyperplane
``coef[i] · x + intercept[i] = 0``. Covariate (and prior) bags share one fixed
boundary, so all rows are identical; **concept** bags each carry their own
rotated boundary.

For ``k > 2`` classes the shapes become ``(n_bags, k, n_features)`` and
``(n_bags, k)`` — one weight vector per class — and the label is
``argmax(coef[i] · x + intercept[i])``, which carves the feature space into ``k``
piecewise-linear regions rather than splitting it with a single line.

Stacking shifts
------------------------------------------------------------

Real data rarely shifts in one way only. Pass a **list** to ``shift_type`` to
combine them:

.. code-block:: python

    Xs, ys, prevs = make_quantification(
        n_batches=50, shift_type=["prior", "covariate", "concept"],
        prevalence="uniform", covariate_scale=0.4, concept_strength=0.3,
        random_state=0)

Stacked shifts compose per bag in a fixed order: covariate translates the
features, concept rotates the boundary, the bag is relabelled by that boundary,
and prior finally resamples it to the target prevalence — so every bag can differ
in position, labelling rule **and** class balance at once.

Training data and difficulty
------------------------------------------------------------

Set ``return_train=True`` to also receive a clean training sample (drawn from a
disjoint half of the population at ``train_prevalence``), so you can fit a
quantifier and score it on the shifted bags in a single call:

.. code-block:: python

    X_train, y_train, Xs, ys, prevs = make_quantification(
        n_batches=100, return_train=True, train_prevalence=[0.5, 0.5],
        random_state=0)

How hard the bags are to quantify is governed by the underlying problem:
``class_sep`` (lower is harder), ``flip_y`` (more label noise), ``n_features``
(more noise dimensions) and ``batch_size`` (smaller bags are noisier).

.. seealso::

   - :ref:`sphx_synthetic_intro` and the surrounding gallery section visualise
     every option above with plots.
   - :func:`make_quantification` — the full parameter reference.
