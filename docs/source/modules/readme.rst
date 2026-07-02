.. _readme_methods:

.. currentmodule:: mlquantify.readme

==============
ReadMe Methods
==============

The ReadMe methods (Hopkins & King, 2010; Jerzak, King & Strezhnev, 2022)
estimate class prevalences **directly from features, with no classifier at
all**. Developed for automated content analysis in the social sciences (and,
before that, verbal autopsy in epidemiology — King & Lu, 2008), they are
fully non-aggregative: there is no per-instance prediction step to aggregate
or correct.

.. contents:: Contents
   :local:
   :depth: 2

----

The accounting identity
=======================

Both methods rest on the law of total probability applied to the feature
distribution:

.. math::

   P(S) = \sum_{j=1}^{J} P(S \mid D = j) \, P(D = j)
   \qquad\Longleftrightarrow\qquad
   P(S) = P(S \mid D) \, P(D)

where :math:`S` is a feature representation of a document and :math:`D` its
category. :math:`P(S)` is observable on the **unlabeled** (test) set and
:math:`P(S \mid D)` on the **labeled** (training) set; the class prevalences
:math:`P(D)` are the unknown "regression coefficients", recovered by least
squares constrained to the probability simplex.

The only assumption linking the two samples is the stability of the
class-conditional feature distribution,
:math:`P(S \mid D)_{\text{labeled}} = P(S \mid D)_{\text{unlabeled}}` —
which follows the causal direction of the data-generating process (the
category produces the words, not the reverse). Crucially, **neither**
:math:`P(D)` **nor** :math:`P(S)` needs to match across the two sets, so the
methods tolerate arbitrary prior shift by construction.

ReadMe — profile tabulation over feature subsets
================================================

:class:`ReadMe` works with **binary** features (word-stem indicators). Since
tabulating the joint distribution of :math:`K` features has :math:`2^K`
cells, it repeatedly draws small random subsets of ``subset_size`` features
(the papers use 5–25), tabulates the subset-profile distributions, solves the
constrained least squares in each subset, and averages the estimates — a form
of kernel smoothing that keeps the estimator approximately unbiased while
making it tractable (King & Lu, 2008).

Continuous features are binarized at the labeled-set median by default
(``binarize='auto'``); features are drawn into subsets with probability
proportional to their variance (``variance_weighting=True``), following the
reference implementation.

.. code-block:: python

   from mlquantify.readme import ReadMe

   q = ReadMe(n_subsets=50, subset_size=15, random_state=42, n_jobs=-1)
   q.fit(X_train, y_train)      # labeled set: tabulates P(S|D)
   print(q.predict(X_test))     # unlabeled set: tabulates P(S), solves P(D)

**Limitation.** When the features discriminate the categories weakly, the
estimate shrinks toward the labeled-set proportions (Proposition 3 of Jerzak
et al., 2022) — the motivation for ReadMe2.

ReadMe2 — learned projections and matching
==========================================

:class:`ReadMe2` (requires PyTorch; ``pip install mlquantify[neural]``)
replaces binary profiles with **continuous** features (e.g. word-embedding
summaries) and improves the identity's conditioning in two ways:

1. **Learned projections.** Per bootstrap iteration, a linear-softsign
   projection of the standardized features is optimized by SGD on
   category-balanced labeled batches to jointly maximise *category
   distinctiveness* (separation of the per-class projected means — low bias)
   and *feature distinctiveness* (diverse, decorrelated projections — low
   variance). The estimation then uses conditional means,
   :math:`E[\tilde{S}] = E[\tilde{S} \mid D]\,P(D)`.
2. **Matching.** Labeled documents are matched to the unlabeled set by
   k-nearest neighbours in the projected space before computing
   :math:`E[\tilde{S} \mid D]`, re-weighting the labeled set toward the
   region the unlabeled documents occupy — which reduces bias under
   *semantic change* (drift in :math:`P(S \mid D)`).

Estimates are averaged over ``n_boot`` independent projections (and, within
each, over ``n_boot_match`` matching rounds).

.. code-block:: python

   from mlquantify.readme import ReadMe2

   q = ReadMe2(n_boot=15, sgd_iters=500, random_state=42)
   q.fit(X_train, y_train)
   print(q.predict(X_test))

   # the reference's "no matching" variant
   q = ReadMe2(matching=False, random_state=42).fit(X_train, y_train)

When to use ReadMe methods
==========================

- When hand-labeled data is scarce, non-random, or from a different time
  period than the target set — the setting these methods were designed for.
- When no reliable classifier can be trained but the class-conditional
  feature distributions are stable.
- Prefer :class:`ReadMe2` whenever features are continuous or the categories
  are weakly separated; prefer :class:`ReadMe` for genuinely binary
  indicator features and when torch is unavailable.

References
==========

.. dropdown:: References

   - Hopkins, D. J., & King, G. (2010). A Method of Automated Nonparametric
     Content Analysis for Social Science. *American Journal of Political
     Science*, 54(1), 229–247.
   - Jerzak, C. T., King, G., & Strezhnev, A. (2022). An Improved Method of
     Automated Nonparametric Content Analysis for Social Science.
     *Political Analysis*, 31(1), 42–58.
   - King, G., & Lu, Y. (2008). Verbal Autopsy Methods with Multiple Causes
     of Death. *Statistical Science*, 23(1), 78–91.

.. seealso::

   :ref:`non_aggregative_quantification` for the other classifier-free
   methods (HDx, GHDx).
