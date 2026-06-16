.. _cython_acceleration:

.. currentmodule:: mlquantify.matching

====================
Cython Acceleration
====================

The inner loop of histogram distribution matching (:class:`DyS`, :class:`HDy`,
:class:`HDx`) is accelerated by an **optional compiled Cython kernel**. It is a
pure performance optimisation: the estimates are identical with or without it,
and if the compiled extension is not available the library transparently falls
back to a pure-Python implementation.

.. contents:: Contents
   :local:
   :depth: 2

----

What is the "histogram sweep"?
==============================

A histogram matching quantifier represents each class by a normalised histogram
of classifier scores. For a binary problem it stores the positive and negative
class histograms :math:`H_+` and :math:`H_-`, and estimates the positive
prevalence :math:`\alpha` as the **mixing proportion** whose mixed histogram is
closest to the test histogram :math:`H_U`:

.. math::

   \hat{\alpha} = \arg\min_{\alpha \in [0, 1]}
   \; D\!\left(\, \alpha H_+ + (1 - \alpha) H_- ,\; H_U \,\right)

Finding that :math:`\alpha` means evaluating the distance :math:`D` at many
candidate values of :math:`\alpha` — a **sweep** over the interval (a grid
search, or a ternary search since the objective is unimodal). Each evaluation:

1. builds the mixture ``alpha * Hpos + (1 - alpha) * Hneg``,
2. normalises it,
3. computes a distance (Hellinger / Topsøe / prob-symm) against the test histogram.

For :class:`HDy`/:class:`DyS` this sweep is repeated over **several bin counts**
and the per-bin estimates are aggregated by their median. So a single ``predict``
performs **hundreds to thousands of these tiny distance evaluations**, each on
short histogram vectors (typically 8–110 bins).

----

What Cython does here
=====================

Those thousands of evaluations are the hot path — and because each one works on a
*tiny* array, the cost is dominated by Python/numpy **per-call overhead**
(``np.asarray``, ufunc dispatch, temporary allocation), not by the actual
arithmetic.

The Cython kernel (``mlquantify.matching._histogram_sweep``) collapses the whole
``mixture → normalise → distance → search`` into a **single compiled C function**:

- typed, contiguous ``float64`` memory views (no Python objects in the loop);
- the unimodal search (ternary or grid) runs **inside the C function**, so the
  thousands of distance evaluations never cross back into Python;
- the inner loop is ``nogil`` — pure C, no interpreter overhead.

The result is bit-for-bit the same :math:`\alpha`, computed without the
per-iteration Python overhead.

----

Where it is more efficient — speed and memory
=============================================

**Speed.** The kernel removes the Python/numpy call overhead that dominates the
sweep over tiny arrays, and keeps the search in compiled C. In practice it roughly
**halves** ``DyS``/``HDy`` ``predict`` time relative to the generic Python solver
(more on a fine bin grid or large test sets), and the gap grows with the number of
bin counts swept.

**Memory.** This is where the kernel helps most relative to the pure-Python path:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Path
     - Allocation behaviour
   * - generic Python solver
     - Allocates a **fresh** mixture array (and normalised copies) on **every**
       :math:`\alpha` evaluation — thousands of small temporary arrays per
       ``predict``, all churned through the allocator and garbage collector.
   * - Cython sweep kernel
     - Allocates **one** scratch buffer of length ``n_bins`` and **reuses it** for
       the entire sweep — ``O(n_bins)`` working memory, zero per-iteration
       allocation, cache-friendly contiguous access.

So the kernel turns a stream of thousands of tiny allocations into a single
reused buffer, which both cuts allocator/GC pressure and keeps the working set in
cache.

----

Enabling, checking and toggling
===============================

The kernel is used **automatically** when the compiled extension is present (it
ships in the binary wheels on PyPI). Everything still works without it — just
slower — via the pure-Python fallback.

.. code-block:: python

   import mlquantify.matching._histogram as H

   H._HAS_FAST_KERNEL     # True if the compiled extension is built/imported
   H.USE_SWEEP_KERNEL     # runtime on/off switch (default True)

   # force the generic Python solver (e.g. for debugging / benchmarking)
   H.USE_SWEEP_KERNEL = False
   # ... and back on
   H.USE_SWEEP_KERNEL = True

If the compiled kernel is not built, the first histogram-matching ``predict``
emits a one-time :class:`PerformanceWarning`. To build it from a source checkout::

   pip install -e .            # compiles mlquantify/matching/_histogram_sweep.pyx

.. note::

   The kernel only changes *speed and memory*, never the result: the estimated
   prevalence is identical (to floating-point tolerance) whether it runs the
   compiled kernel, the pure-Python sweep, or the generic solver.

.. seealso::

   :ref:`distribution_matching` for the methods that use the sweep, and
   :ref:`representations` for the :class:`~mlquantify.representations.HistogramRepresentation`
   it operates on.
