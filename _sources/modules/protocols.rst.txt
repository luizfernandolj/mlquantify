.. _quantification_protocols:

.. currentmodule:: mlquantify.model_selection

==============================
Protocols for Quantification
==============================

Evaluating a quantifier on a single test set is misleading — the test
prevalence is fixed, so you only see performance at one operating point.
Quantification protocols address this by generating **many test batches**
with varying prevalences from the same data, giving a fuller picture of
method behaviour across the entire prevalence spectrum.

.. admonition:: Why protocols matter

   A quantifier that looks excellent at 50/50 prevalence may fail badly at
   5/95. Forman (2005) noted that the choice of evaluation protocol is as
   important as the choice of method. Standard practice in quantification
   research is to evaluate across a grid of prevalences (APP) and report the
   mean error over all samples.

.. contents:: Contents
   :local:
   :depth: 2

----

APP — Artificial Prevalence Protocol
======================================

:class:`APP` is the most widely used evaluation protocol. It draws samples
from the test set at each prevalence in a uniform grid
:math:`\{0, \frac{1}{n-1}, \frac{2}{n-1}, \ldots, 1\}` for the positive
class, repeating each prevalence ``repeats`` times.

**Why it is standard:** APP ensures every method is evaluated at many
prevalence values, not just the natural one. It exposes systematic biases
(e.g. methods that only work near 50/50) and gives a fair cross-method
comparison. González et al. (2017) review papers routinely use APP as the
evaluation backbone.

Parameters
----------

.. list-table::
   :widths: 22 15 63
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``batch_size``
     - required
     - Number of instances per test sample. Larger batches give more stable
       prevalence estimates but require a larger test set. A typical choice is
       100–500.
   * - ``n_prevalences``
     - ``21``
     - Number of equally-spaced prevalence points from ``min_prev`` to
       ``max_prev``. ``21`` gives a step of 0.05 (i.e. 0%, 5%, 10%, …, 100%).
   * - ``repeats``
     - ``10``
     - How many independent samples to draw at each prevalence level. More
       repeats reduce variance in the average error estimate. Use ≥ 5 for
       reliable results.
   * - ``min_prev``
     - ``0.0``
     - Minimum positive class prevalence in the grid. Leave at 0 to include
       the all-negative case.
   * - ``max_prev``
     - ``1.0``
     - Maximum positive class prevalence. Leave at 1 to include the
       all-positive case.
   * - ``random_state``
     - ``None``
     - Seed for reproducible sampling.

.. figure:: ../images/app_protocol.png
   :align: center
   :width: 90%
   :alt: APP vs NPP protocol comparison

   *Left: APP generates test samples at every point on a regular prevalence grid
   (blue dots), giving systematic coverage from 0% to 100% positive class.
   Right: NPP draws random sub-samples that cluster near the natural training
   prevalence (~50%), providing realistic but narrower coverage.*

Examples
--------

Standard evaluation loop:

.. code-block:: python

   from mlquantify.model_selection import APP
   from mlquantify.metrics import MAE
   from mlquantify.utils import get_prev_from_labels
   from mlquantify.likelihood import EMQ
   from sklearn.linear_model import LogisticRegression
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split
   import numpy as np

   X, y = make_classification(n_samples=2000, weights=[0.7, 0.3],
                              random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.5, random_state=42)

   q = EMQ(LogisticRegression())
   q.fit(X_train, y_train)

   protocol = APP(batch_size=100, n_prevalences=21, repeats=10,
                  random_state=42)
   errors = []
   for idx in protocol.split(X_test, y_test):
       X_sample, y_sample = X_test[idx], y_test[idx]
       true_prev = get_prev_from_labels(y_sample)
       pred_prev = q.predict(X_sample)
       errors.append(MAE(true_prev, pred_prev))

   print(f"Mean MAE over {len(errors)} samples: {np.mean(errors):.4f}")

Comparing multiple quantifiers:

.. code-block:: python

   from mlquantify.counting import CC, PCC
   from mlquantify.likelihood import EMQ
   from mlquantify.matching import DyS
   from sklearn.linear_model import LogisticRegression

   quantifiers = {
       'CC':  CC(LogisticRegression()),
       'PCC': PCC(LogisticRegression()),
       'EMQ': EMQ(LogisticRegression()),
       'DyS': DyS(LogisticRegression()),
   }

   for name, q in quantifiers.items():
       q.fit(X_train, y_train)

   protocol = APP(batch_size=100, n_prevalences=21, repeats=10,
                  random_state=42)

   results = {name: [] for name in quantifiers}
   for idx in protocol.split(X_test, y_test):
       X_s, y_s = X_test[idx], y_test[idx]
       true_prev = get_prev_from_labels(y_s)
       for name, q in quantifiers.items():
           results[name].append(MAE(true_prev, q.predict(X_s)))

   for name, errs in results.items():
       print(f"{name:5s}  MAE={np.mean(errs):.4f}")

----

NPP — Natural Prevalence Protocol
===================================

:class:`NPP` draws random sub-samples from the test set without altering
the natural class distribution. Each sample has a slightly different
prevalence due to random variation, but no artificial manipulation is
performed.

**Why it exists:** NPP evaluates quantifiers under *real* prevalence
variation — how they perform when deployed on random sub-populations drawn
from the same underlying distribution as the test set. It is less controlled
than APP but more realistic.

**Limitation:** Because NPP cannot produce extreme prevalences (e.g. 2%
positive) without a very large test set, it gives a narrower view of method
behaviour than APP.

Parameters
----------

.. list-table::
   :widths: 22 15 63
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``batch_size``
     - required
     - Size of each random sub-sample.
   * - ``n_samples``
     - ``100``
     - Number of random sub-samples to draw.
   * - ``random_state``
     - ``None``
     - Seed for reproducibility.

.. code-block:: python

   from mlquantify.model_selection import NPP
   from mlquantify.utils import get_prev_from_labels

   protocol = NPP(batch_size=100, n_samples=50, random_state=42)
   for idx in protocol.split(X_test, y_test):
       X_s, y_s = X_test[idx], y_test[idx]
       true_prev = get_prev_from_labels(y_s)
       pred_prev = q.predict(X_s)

----

UPP — Uniform Prevalence Protocol
===================================

:class:`UPP` samples prevalence vectors uniformly from the **probability
simplex**. For binary problems it is similar to APP, but for
**multiclass** problems it avoids the combinatorial explosion of sweeping
all class-prevalence combinations independently.

**Why it exists:** For :math:`k` classes, a grid approach like APP grows as
:math:`O(n^{k-1})` which quickly becomes intractable. UPP samples :math:`n`
random vectors from the simplex, covering the multiclass prevalence space
efficiently without a rigid grid. Maletzke et al. (2020) recommend UPP for
multiclass evaluation.

Parameters
----------

.. list-table::
   :widths: 22 15 63
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``batch_size``
     - required
     - Size of each sample.
   * - ``n_prevalences``
     - ``100``
     - Number of prevalence vectors to sample from the simplex.
   * - ``algorithm``
     - ``'uniform'``
     - Sampling algorithm:

       - ``'uniform'`` — samples from the flat Dirichlet distribution
         (:math:`\text{Dir}(\mathbf{1})`). All prevalence combinations are
         equally likely.
       - ``'kraemer'`` — generates prevalences with a fixed step size,
         analogous to APP for multiclass. More systematic, fewer samples.
   * - ``min_prev``
     - ``0.0``
     - Minimum per-class prevalence. Raise (e.g. to ``0.01``) to avoid
       near-zero classes that are hard to sample from small datasets.
   * - ``max_prev``
     - ``1.0``
     - Maximum per-class prevalence.
   * - ``random_state``
     - ``None``
     - Seed.

.. code-block:: python

   from mlquantify.model_selection import UPP
   from mlquantify.utils import get_prev_from_labels
   from sklearn.datasets import make_classification

   X, y = make_classification(n_samples=2000, n_classes=4,
                              n_informative=6, n_redundant=0,
                              random_state=42)
   X_train, X_test = X[:1500], X[1500:]
   y_train, y_test = y[:1500], y[1500:]

   protocol = UPP(batch_size=100, n_prevalences=200, algorithm='uniform',
                  random_state=42)
   errors = []
   for idx in protocol.split(X_test, y_test):
       X_s, y_s = X_test[idx], y_test[idx]
       true_prev = get_prev_from_labels(y_s)
       pred_prev = q.predict(X_s)
       errors.append(MAE(true_prev, pred_prev))

----

Choosing a Protocol
=====================

.. list-table::
   :widths: 15 20 65
   :header-rows: 1

   * - Protocol
     - Problem type
     - Use when
   * - APP
     - Binary
     - **Default for binary problems.** Systematic sweep; standard in
       quantification research. Forman (2005) introduced the concept.
   * - NPP
     - Binary / multiclass
     - You want realistic evaluation under natural prevalence variation.
   * - UPP (uniform)
     - Multiclass
     - **Default for multiclass.** Efficient random coverage of the simplex.
   * - UPP (kraemer)
     - Multiclass
     - You need a deterministic grid equivalent to APP for multiclass.

.. tip::

   Always fix ``random_state`` in protocols when comparing methods so that
   all quantifiers are evaluated on exactly the same test samples.

.. seealso::

   :ref:`quantification_foundations` for a conceptual overview of why
   protocols are necessary. :ref:`model_selection` for hyperparameter
   tuning with ``GridSearchQ``.

Protocols for Quantification
==============================

Quantification protocols are designed to evaluate quantifiers by generating multiple test samples with varying class prevalences. These protocols ensure robust assessment of quantification methods under different distributional shifts.

Experimental evaluation primarily uses two main protocols:

Artificial-Prevalence Protocol (APP)
====================================

The :class:`APP` is the most commonly used protocol, leveraging widely available classification datasets to artificially vary class prevalences in test samples.

- Generates multiple test samples by subsampling the original test set to produce varying class prevalences.
- Simulates prior probability shift (:math:`P_L(Y) \neq P_U(Y)`) while maintaining conditional feature distributions constant.
- Allows creation of extensive test points from a single dataset for thorough evaluation.

**Example**

.. code-block:: python

    from mlquantify.model_selection import APP
    from mlquantify.utils import get_prev_from_labels

    # Initialize protocol
    app = APP(
        batch_size=[100, 200], 
        n_prevalences=5, 
        repeats=3, 
        random_state=42
    )

    for idx in app.split(X_test, y_test):
        X_sample, y_sample = X_test[idx], y_test[idx]
        real_prevalence = get_prev_from_labels(y_sample)
        # Evaluate quantifier on (X_sample, y_sample)


Natural-Prevalence Protocol (NPP)
=================================

The NPP uses naturally occurring prevalence variations by partitioning a large test set into random sub-samples, preserving their inherent class distributions.

- Preserves real-world prevalence distributions without artificial manipulation.
- Provides realistic evaluation of quantifiers but is less common due to data requirements.

**Example**

.. code-block:: python

    from mlquantify.model_selection import NPP
    from mlquantify.utils import get_prev_from_labels

    # Initialize protocol
    npp = NPP(batch_size=100, random_state=42)

    for idx in npp.split(X_test, y_test):
        X_sample, y_sample = X_test[idx], y_test[idx]
        real_prevalence = get_prev_from_labels(y_sample)
        # Evaluate quantifier on (X_sample, y_sample)



Uniform Prevalence Protocol (UPP)
=================================

The :class:`UPP` is a variant of the APP that ensures uniform sampling of class prevalences across the entire range [0, 1].

- Guarantees that all possible prevalence values are equally represented in the test samples.
- Useful for comprehensive evaluation of quantifiers across the full prevalence spectrum.
- Particularly beneficial in multiclass quantification tasks (less computationally intensive).

**Example**

.. code-block:: python

    from mlquantify.model_selection import UPP
    from mlquantify.utils import get_prev_from_labels

    # Initialize protocol
    upp = UPP(
        batch_size=[100, 200], 
        n_prevalences=5, 
        repeats=3, 
        random_state=42
    )

    for idx in upp.split(X_test, y_test):
        X_sample, y_sample = X_test[idx], y_test[idx]
        real_prevalence = get_prev_from_labels(y_sample)
        # Evaluate quantifier on (X_sample, y_sample)


Personalized Prevalence Protocol (PPP)
======================================

The :class:`PPP` is another APP variant that allows users to specify desired class prevalences for generating test samples, since APP sample all possible prevalences uniformly.

- Enables targeted evaluation of quantifiers at specific prevalence levels.
- Useful for scenarios where certain prevalence values are of particular interest.

**Example**

.. code-block:: python

    from mlquantify.model_selection import PPP
    from mlquantify.utils import get_prev_from_labels

    # Initialize protocol with desired prevalences
    ppp = PPP(batch_size=100, prevalences=[0.1, 0.9], repeats=3, random_state=42)

    for idx in ppp.split(X_test, y_test):
        X_sample, y_sample = X_test[idx], y_test[idx]
        real_prevalence = get_prev_from_labels(y_sample)
        # Evaluate quantifier on (X_sample, y_sample)


References
==========

.. [1] Esuli, A., Fabris, A., Moreo, A., & Sebastiani, F. (n.d.). Learning to Quantify The Information Retrieval Series.