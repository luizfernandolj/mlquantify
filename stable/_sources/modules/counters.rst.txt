.. _counters_module:

.. currentmodule:: mlquantify.counting

===========================
Counting-Based Quantifiers
===========================

Counting-based quantifiers are the simplest family of aggregative methods:
train a classifier, apply it to the test set, and count the fraction of
instances assigned to each class. They are cheap, easy to understand, and
serve as essential baselines.

See :ref:`quantification_foundations` for the theoretical background on *why*
counting alone is biased and *when* you need a stronger correction.

.. contents:: Contents
   :local:
   :depth: 2

----

CC — Classify and Count
========================

:class:`CC` is the simplest quantification baseline. It trains a hard
classifier :math:`h` on labelled data :math:`L`, applies it to an unlabelled
set :math:`U`, and counts the proportion of predictions for each class:

.. math::

   \hat{p}^{CC}(c) = \frac{|\{x \in U : h(x) = c\}|}{|U|}

**Why it exists:** CC is the *reference baseline* for every quantification
study. Despite being biased under distributional shift (Forman, 2005), it is
fast, interpretable, and competitive when training and test prevalences are
close. Always include it as a point of comparison.

.. figure:: ../images/cc_bias.png
   :align: center
   :width: 75%
   :alt: CC and PCC prevalence estimation bias

   *A classifier trained on 50/50 balanced data is evaluated on test sets with
   varying true prevalences. CC (red) consistently overestimates at low prevalences
   and underestimates at high ones. PCC (orange) is less biased but still distorted.
   The dashed line is the ideal unbiased estimator.*

.. admonition:: When CC fails

   Suppose a classifier trained on balanced data (50 % positive) achieves
   90 % accuracy, but the test set has only 5 % positives. CC estimates
   :math:`\approx 14\%` positives — nearly 3× the truth — because the
   false positives from the 95 % negatives dominate the count. The bias
   does not vanish with more data; it vanishes only when the distributions
   match.

Parameters
----------

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``estimator``
     - ``None``
     - Any scikit-learn-compatible classifier with ``fit`` and ``predict``
       methods. If ``None``, skip fitting and call :meth:`aggregate` directly
       with pre-computed hard labels.
   * - ``threshold``
     - ``0.5``
     - Decision threshold applied to soft scores when the estimator exposes
       ``predict_proba``. Values above the threshold are labelled positive.
       Rarely needs tuning here — use :class:`TAC`, :class:`T50`, or
       :class:`TMAX` if threshold selection matters for your problem.

Examples
--------

Basic usage:

.. code-block:: python

   from mlquantify.counting import CC
   from sklearn.linear_model import LogisticRegression
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   X, y = make_classification(n_samples=1000, weights=[0.8, 0.2],
                              random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42)

   q = CC(LogisticRegression())
   q.fit(X_train, y_train)
   print(q.predict(X_test))
   # {0: 0.82, 1: 0.18}

Using :meth:`aggregate` with pre-computed hard labels:

.. code-block:: python

   import numpy as np
   from mlquantify.counting import CC

   q = CC()
   hard_labels = np.array([0, 0, 1, 0, 1, 1, 0, 0, 0, 1])
   print(q.aggregate(hard_labels))
   # {0: 0.6, 1: 0.4}

.. warning::

   CC is consistently biased when the test class distribution differs from
   training. For any real deployment scenario with distribution shift, use
   :class:`ACC`, :class:`~mlquantify.likelihood.EMQ`, or a
   distribution-matching method instead.

----

PCC — Probabilistic Classify and Count
========================================

:class:`PCC` replaces hard decisions with **posterior probabilities** and
estimates prevalence as their average over the test set:

.. math::

   \hat{p}^{PCC}(c) = \frac{1}{|U|} \sum_{x \in U} P(Y = c \mid x)

where :math:`P(Y=c \mid x)` is the soft output of a probabilistic classifier.

**Why it exists:** PCC smooths out the hard-thresholding noise in CC. By
averaging probabilities instead of counting hard predictions, it reduces
variance and often reduces bias too — especially when the classifier is
well-calibrated. However, Forman (2005) showed that uncalibrated posteriors
under prior-probability shift are still biased, so PCC is not a complete
solution. It is the best quick upgrade from CC.

Parameters
----------

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``estimator``
     - ``None``
     - A probabilistic classifier with ``fit`` and ``predict_proba`` methods.
       Required for soft probability outputs. If ``None``, call
       :meth:`aggregate` directly with a pre-computed probability matrix of
       shape ``(n_samples, n_classes)``.

Examples
--------

.. code-block:: python

   from mlquantify.counting import PCC
   from sklearn.ensemble import RandomForestClassifier

   q = PCC(RandomForestClassifier(n_estimators=100, random_state=42))
   q.fit(X_train, y_train)
   print(q.predict(X_test))
   # {0: 0.80, 1: 0.20}

Using :meth:`aggregate` with pre-computed posteriors:

.. code-block:: python

   import numpy as np
   from mlquantify.counting import PCC

   # Shape: (n_samples, n_classes)
   proba = np.array([[0.9, 0.1],
                     [0.3, 0.7],
                     [0.6, 0.4]])
   q = PCC()
   print(q.aggregate(proba))
   # {0: 0.6, 1: 0.4}

----

GACC — Generalised Adjusted Classify and Count
================================================

:class:`GACC` extends the binary ACC correction to **native multiclass**
problems. During training, it builds a confusion matrix :math:`M` via
cross-validation (where :math:`M_{ij}` is the fraction of class-:math:`i`
samples predicted as class :math:`j`). At prediction time it solves the
constrained linear system:

.. math::

   \hat{p} = \arg\min_{\hat{p} \in \Delta^{k-1}} \| M\hat{p} - \hat{c} \|

where :math:`\hat{c}` is the vector of CC proportions on the test set.

**Why it exists:** Binary ACC cannot handle more than two classes directly,
because the 2×2 confusion-matrix correction becomes underdetermined for
:math:`k>2`. GACC provides the proper multiclass generalisation via
constrained optimisation, as introduced by Firat (2016).

Parameters
----------

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``estimator``
     - ``None``
     - Hard-prediction classifier (``fit`` + ``predict``). Does not need to be
       probabilistic.
   * - ``loss``
     - ``'ls'``
     - Loss used to solve the linear system. ``'ls'`` (least squares) is the
       standard choice. ``'l1'`` adds robustness against outlier confusion-matrix
       entries at the cost of a slower solve.
   * - ``solver``
     - ``'slsqp'``
     - Optimisation algorithm. ``'slsqp'`` handles the simplex constraint
       (:math:`\hat{p} \ge 0`, :math:`\sum \hat{p}=1`) natively and is the
       recommended choice.
   * - ``cv``
     - ``None`` (→ 5)
     - Cross-validation folds for building the confusion matrix. More folds give
       a more accurate estimate of the confusion matrix at the cost of extra
       training time. 5 is a good default; use 10 on larger datasets.
   * - ``stratified``
     - ``True``
     - Stratify folds to ensure every class appears in every fold — essential
       when classes are imbalanced. Leave ``True`` unless you have a specific
       reason not to.
   * - ``shuffle``
     - ``False``
     - Whether to shuffle data before splitting. Set ``True`` if the dataset has
       temporal ordering that could make consecutive samples very similar.
   * - ``random_state``
     - ``None``
     - Seed for reproducibility of the cross-validation split.

Examples
--------

Three-class quantification:

.. code-block:: python

   from mlquantify.counting import GACC
   from sklearn.svm import SVC
   from sklearn.datasets import make_classification

   X, y = make_classification(n_samples=600, n_classes=3,
                              n_informative=5, n_redundant=0,
                              random_state=42)
   X_train, X_test = X[:450], X[450:]
   y_train, y_test = y[:450], y[450:]

   q = GACC(SVC(), cv=5, stratified=True)
   q.fit(X_train, y_train)
   print(q.predict(X_test))
   # {0: 0.33, 1: 0.34, 2: 0.33}

.. note::

   GACC needs at least ``cv`` training samples per class. If a class is very
   rare, reduce ``cv`` or use stratified splits to prevent empty folds.

----

GPACC — Generalised Probabilistic Adjusted Classify and Count
==============================================================

:class:`GPACC` is the soft analogue of :class:`GACC`. Instead of a hard
confusion matrix, it builds a **soft confusion matrix** from posterior
probabilities:

.. math::

   M_{ij} = \frac{1}{|L_i|} \sum_{x \in L_i} P(Y = c_j \mid x)

where :math:`L_i` is the set of training examples with true class :math:`c_i`.
The prevalence is then estimated by solving the same constrained system as GACC.

**Why it exists:** Soft matrices preserve more information than hard ones (no
thresholding artefacts). GPACC is usually more accurate than GACC when the
classifier is well-calibrated and typically outperforms plain PCC. It is the
recommended native-multiclass counting method.

Parameters
----------

Same as :class:`GACC`. The ``estimator`` must support ``predict_proba``.

Examples
--------

.. code-block:: python

   from mlquantify.counting import GPACC
   from sklearn.linear_model import LogisticRegression
   from sklearn.datasets import make_classification

   X, y = make_classification(n_samples=600, n_classes=4,
                              n_informative=6, n_redundant=0,
                              random_state=42)
   X_train, X_test = X[:450], X[450:]
   y_train, y_test = y[:450], y[450:]

   q = GPACC(LogisticRegression(), cv=5)
   q.fit(X_train, y_train)
   print(q.predict(X_test))
   # {0: 0.25, 1: 0.26, 2: 0.24, 3: 0.25}

----

Choosing Among CC, PCC, GACC, and GPACC
==========================================

.. list-table::
   :widths: 18 15 18 20 14
   :header-rows: 1

   * - Method
     - Shift correction
     - Native multiclass
     - Needs ``predict_proba``
     - Extra fit cost
   * - CC
     - ✗
     - ✓
     - ✗
     - None
   * - PCC
     - ✗
     - ✓
     - ✓
     - None
   * - GACC
     - ✓ (linear)
     - ✓
     - ✗
     - CV folds
   * - GPACC
     - ✓ (linear)
     - ✓
     - ✓
     - CV folds

**Practical recommendation:**

- Use **CC** only as a sanity-check baseline.
- Use **GPACC** as your primary counting-family method: the soft matrix
  and simplex constraint make it noticeably better than CC/PCC under shift.
- If probabilities are unavailable, fall back to **GACC**.
- For large distributional shifts, graduate to :class:`~mlquantify.likelihood.EMQ`
  or a distribution-matching method from :mod:`mlquantify.matching`.

.. seealso::

   :ref:`counting` for threshold-adjustment methods (:class:`ACC`,
   :class:`TAC`, :class:`TX`, …) which offer stronger binary-specific correction.

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

   from mlquantify.counting import CC
   from sklearn.linear_model import LogisticRegression
   import numpy as np

   X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
   q = CC(estimator=LogisticRegression())
   q.fit(X, y)
   q.predict(X)
   # -> {0: 0.47, 1: 0.53}


.. note::
    :class:`CC` is fast and simple, but when class proportions in the test set differ from the training set, its estimates can become biased or inaccurate.



Probabilistic Classify and Count  
================================

The **Probabilistic Classify and Count** or :class:`PCC` variant uses the *predicted probabilities* from a soft classifier instead of hard labels.  
This makes it less sensitive to uncertain predictions.

[Plot Idea: A plot comparing probabilities per sample and their averaged mean per class]

**Example**

.. code-block:: python

   from mlquantify.counting import PCC
   from sklearn.linear_model import LogisticRegression
   import numpy as np

   X, y = np.random.randn(100, 5), np.random.randint(0, 2, 100)
   q = PCC(estimator=LogisticRegression())
   q.fit(X, y)
   q.predict(X)
   # -> {0: 0.45, 1: 0.55}

CC and PCC both often underestimate or overestimate the true prevalence when there is distribution shift (also known as "dataset shift").