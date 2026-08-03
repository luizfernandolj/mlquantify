.. _explicit_loss_minimization:

.. currentmodule:: mlquantify.elm

==========================
Explicit Loss Minimization
==========================

Most aggregative quantifiers train an off-the-shelf classifier and correct
its counts afterwards. Explicit Loss Minimization (ELM) methods take the
opposite route: they **train the classifier itself to minimize a
quantification-oriented loss**, so that plain Classify & Count over its
predictions is already a good prevalence estimator. The machinery is
Joachims' :math:`SVM^{\Delta}_{multi}` — the structural SVM behind
``svmperf`` — which can optimize *any* loss computed from the contingency
table of the whole sample. mlquantify ships a **pure-Python
reimplementation** of it (:class:`MultivariateLossSVM`), so no external
binary is needed.

.. contents:: Contents
   :local:
   :depth: 2

----

The multivariate SVM
====================

A standard SVM sums example-wise hinge losses, which upper-bounds the error
*rate* — a loss that decomposes over examples. Quantification losses do
not decompose: :math:`|FP - FN|` depends on the joint prediction over the
whole sample. Joachims (2005) reformulates learning as predicting the
entire label *tuple* at once:

.. math::

   \min_{w,\,\xi \ge 0}\ \tfrac{1}{2}\|w\|^2 + C\,\xi
   \quad \text{s.t.} \quad
   w^\top[\Psi(\bar{x},\bar{y}) - \Psi(\bar{x},\bar{y}')] \ge
   \Delta(\bar{y}',\bar{y}) - \xi \quad \forall \bar{y}'

with :math:`\Psi(\bar{x},\bar{y}') = \sum_i y_i' x_i`. There is one
constraint per possible labeling, but a cutting-plane algorithm adds only
the *most violated* one per iteration — found in a single vectorized pass
over all :math:`O(n^2)` contingency tables — and converges after tens of
iterations. At the solution, the slack :math:`\xi` upper-bounds the
training loss :math:`\Delta`.

The Q-measure — reliable quantifiers
====================================

Any hyperplane with :math:`FP = FN` on the training sample is a *perfect
quantifier* there — including useless ones (Barranquero et al., 2015,
Fig. 1: a plane separating an unrelated class can balance the errors
perfectly). Pure quantification losses therefore admit degenerate optima.
The **Q-measure** removes them by mixing in classification reliability:

.. math::

   Q_\beta = (1+\beta^2)\,
   \frac{\text{recall} \cdot \text{NAS}}{\beta^2\,\text{recall} + \text{NAS}},
   \qquad
   \text{NAS} = 1 - \frac{|FN - FP|}{\max(P, N)}

Recall acts as a *hook*: it forces coherent predictions on the positive
class (low FN), and NAS keeps the residual errors compensated. ``beta``
trades the two off (:math:`\beta \to 0` recovers recall,
:math:`\beta \to \infty` recovers NAS; the paper analyses 0.5, 1 and 2).

The methods
===========

All quantifiers below are Classify & Count over the loss-optimized SVM
(the classifier is intrinsic — no ``estimator`` parameter). They are
binary; multiclass problems are decomposed automatically via One-vs-Rest.

- :class:`SVMQ` — the Q-measure loss (Barranquero et al., 2015).
- :class:`SVMKLD` / :class:`SVMNKLD` — (normalised) Kullback–Leibler
  divergence between true and predicted prevalences (Esuli & Sebastiani,
  2015).
- :class:`SVMAE` / :class:`SVMRAE` — (relative) absolute prevalence error.
- :class:`ELM` — the generic base: any of the losses above, ``'error'``
  (recovers a standard linear SVM — useful as a controlled baseline),
  ``'f1'``, or **your own callable** ``loss(a, b, n_pos, n_neg)`` over
  contingency-table grids.

Examples
--------

.. code-block:: python

   from mlquantify.elm import SVMQ
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   X, y = make_classification(n_samples=2000, weights=[0.75, 0.25],
                              random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42)

   q = SVMQ(C=1.0, beta=1.0)
   q.fit(X_train, y_train)
   print(q.predict(X_test))

The raw learner is public and composes with any aggregative quantifier —
e.g. an adjusted count on top of the Q-optimized hyperplane:

.. code-block:: python

   from mlquantify.counting import ACC
   from mlquantify.elm import MultivariateLossSVM

   q = ACC(estimator=MultivariateLossSVM(loss='q'))
   q.fit(X_train, y_train)

A custom loss is a function of the contingency-table grids:

.. code-block:: python

   import numpy as np
   from mlquantify.elm import ELM

   def balanced_error(a, b, n_pos, n_neg):
       c = n_pos - a
       return 100.0 * 0.5 * (c / n_pos + b / n_neg)

   q = ELM(loss=balanced_error).fit(X_train, y_train)

Practical guidance
------------------

- **When it shines:** imbalanced training sets, where an accuracy-trained
  classifier skews toward the majority class and its counts inherit that
  bias. The Q/KLD-trained hyperplanes balance the errors at the source.
- ``C`` trades regularization against the (0–100 scaled) loss; tune it
  with :class:`~mlquantify.model_selection.GridSearchQ` together with
  ``beta`` for :class:`SVMQ`.
- ``tol`` and ``max_iter`` control the cutting-plane loop; defaults
  converge in tens of iterations. The constraint search materialises an
  ``(P+1, N+1)`` grid per iteration, so memory grows quadratically with
  the training-set size.
- The model is a **linear** SVM without intercept (append a constant
  feature if one is needed), matching how the ELM methods are used in the
  literature.

References
==========

.. dropdown:: References

   - Barranquero, J., Díez, J., & del Coz, J. J. (2015).
     Quantification-oriented learning based on reliable classifiers.
     *Pattern Recognition*, 48(2), 591–604.
   - Joachims, T. (2005). A Support Vector Method for Multivariate
     Performance Measures. *ICML*, pp. 377–384.
   - Joachims, T. (2006). Training Linear SVMs in Linear Time. *KDD*.
   - Esuli, A., & Sebastiani, F. (2015). Optimizing Text Quantifiers for
     Multivariate Loss Functions. *ACM TKDD*, 9(4), 27.

.. seealso::

   :ref:`counters_module` for CC and ACC (which ELM composes with).
   :ref:`quantification_trees` for the other learning-phase quantification
   family (decision trees with error-balancing splits).
