.. _examples:

========
Examples
========

A gallery of short, self-contained scripts that illustrate how to use
``mlquantify`` in practice. Every example renders its full source code above the
figure it produces — click the copy button in the top-right corner of a code
block to run it as-is.

The examples are grouped by theme. If you are new to quantification, start with
:ref:`sphx_quant_intro` and :ref:`sphx_cc_under_shift`.


Quantification basics
---------------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Introduction to quantification
        :link: sphx_quant_intro
        :link-type: ref
        :text-align: center

        Fit your first quantifier, predict class prevalence on a test sample,
        and compare it against the ground truth.

    .. grid-item-card:: Why counting fails under prior shift
        :link: sphx_cc_under_shift
        :link-type: ref
        :text-align: center

        The motivating example for the whole field: plain Classify & Count is
        biased when the test prevalence drifts away from training.


Comparing methods
------------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Comparing quantifiers with diagonal plots
        :link: sphx_method_comparison
        :link-type: ref
        :text-align: center

        Run several quantifiers through an Artificial Prevalence Protocol and
        read off their bias and variance from true-vs-predicted diagonal plots.

    .. grid-item-card:: Distribution matching, step by step
        :link: sphx_distribution_matching
        :link-type: ref
        :text-align: center

        Visualise the mechanism behind HDy and DyS: match the test score
        histogram with a mixture of the class-conditional histograms.

    .. grid-item-card:: EMQ and the EM prior correction
        :link: sphx_emq_convergence
        :link-type: ref
        :text-align: center

        Watch the Expectation-Maximisation loop iteratively correct the prior
        until the adjusted prevalence converges to the true one.

    .. grid-item-card:: Neural quantifiers (HistNetQ & GMNet)
        :link: sphx_neural_quantifiers
        :link-type: ref
        :text-align: center

        Train two symmetric neural quantifiers end-to-end on bags of instances
        and read their bias and variance off diagonal plots.

    .. grid-item-card:: QuaNet: deep quantification with LSTMs
        :link: sphx_quanet
        :link-type: ref
        :text-align: center

        Put a recurrent correction network on top of a regular classifier and
        watch it undo the classifier's counting bias on shifted bags.

    .. grid-item-card:: Quantification trees and forests
        :link: sphx_quantification_trees
        :link-type: ref
        :text-align: center

        Grow decision trees whose splits balance false positives against false
        negatives, and compare them with a standard CART under shift.

    .. grid-item-card:: Quantifying without a classifier (ReadMe)
        :link: sphx_readme_methods
        :link-type: ref
        :text-align: center

        Estimate prevalences directly from features with the ReadMe accounting
        identity — no per-document prediction step at all.

    .. grid-item-card:: Training the classifier for quantification (SVM(Q))
        :link: sphx_elm
        :link-type: ref
        :text-align: center

        Train an SVM whose loss is the Q-measure itself and watch its raw
        counts stay honest where an error-trained SVM tilts.


Evaluation and protocols
-------------------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Evaluation protocols (APP, NPP, UPP)
        :link: sphx_protocols
        :link-type: ref
        :text-align: center

        See the different test distributions that each sampling protocol
        produces, and when to reach for each one.

    .. grid-item-card:: Robustness to prior-probability shift
        :link: sphx_error_by_shift
        :link-type: ref
        :text-align: center

        Plot quantification error as a function of how far the test prevalence
        drifts from training, and compare methods on robustness.


Model selection and uncertainty
--------------------------------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Tuning a quantifier with GridSearchQ
        :link: sphx_grid_search
        :link-type: ref
        :text-align: center

        Select hyper-parameters with a quantification-aware grid search driven
        by an evaluation protocol.

    .. grid-item-card:: Confidence intervals and regions
        :link: sphx_confidence_regions
        :link-type: ref
        :text-align: center

        Turn a point prevalence estimate into a bootstrap confidence interval
        and, for three classes, a confidence ellipse on the simplex.

    .. grid-item-card:: Calibrating classifier posteriors
        :link: sphx_calibration
        :link-type: ref
        :text-align: center

        Fix an over-confident classifier with temperature / vector scaling and
        watch the reliability diagram snap onto the diagonal — the posterior fix
        that EMQ relies on.


Multiclass
----------

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Multiclass quantification
        :link: sphx_multiclass
        :link-type: ref
        :text-align: center

        Quantify a three-class problem and inspect the per-class accuracy and
        the joint uncertainty on the probability simplex.


Real-world datasets
-------------------

Load well-known quantification benchmarks with the
:mod:`~mlquantify.datasets` fetchers, then score quantifiers on them.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Loading real-world datasets
        :link: sphx_real_datasets_loading
        :link-type: ref
        :text-align: center

        Fetch and cache benchmark datasets as a Bunch or an ``(X, y)`` tuple,
        as NumPy arrays or pandas frames.

    .. grid-item-card:: Evaluating a quantifier on real data
        :link: sphx_real_datasets_evaluation
        :link-type: ref
        :text-align: center

        Use a fetcher's built-in protocol to draw test bags with known
        prevalence and score a quantifier against them.


Synthetic datasets with ``make_quantification``
-----------------------------------------------

Build controlled experiments with
:func:`~mlquantify.datasets.make_quantification`: generate labelled bags under
prior-probability shift, see how the data and its prevalence behave, then
benchmark quantifiers on it with the true prevalences in hand.

.. grid:: 1 1 2 2
    :gutter: 3

    .. grid-item-card:: Visualizing synthetic data
        :link: sphx_synthetic_intro
        :link-type: ref
        :text-align: center

        Plot a single bag in two dimensions, coloured by class, to see what the
        generator produces.

    .. grid-item-card:: Prior shift, bag by bag
        :link: sphx_synthetic_shift
        :link-type: ref
        :text-align: center

        Watch the class clusters keep their shape while their balance shifts
        from bag to bag.

    .. grid-item-card:: Controlling prevalence variability
        :link: sphx_synthetic_prevalence
        :link-type: ref
        :text-align: center

        Compare the ``prevalence`` strategies — uniform, grid, natural and
        Dirichlet — and the ``concentration`` knob.

    .. grid-item-card:: Class separability and label noise
        :link: sphx_synthetic_difficulty
        :link-type: ref
        :text-align: center

        Tune how hard the problem is with ``class_sep`` and ``flip_y``, and see
        the effect on quantification error.

    .. grid-item-card:: Covariate and concept shift
        :link: sphx_synthetic_shift_types
        :link-type: ref
        :text-align: center

        Generate the other two kinds of dataset shift, see how each looks, and
        why the best quantifier depends on the shift.

    .. grid-item-card:: Benchmarking quantifiers
        :link: sphx_synthetic_quantifiers
        :link-type: ref
        :text-align: center

        Fit on a training sample, predict many shifted bags, and score methods
        directly against the known prevalences.


.. toctree::
   :hidden:

   plot_quantification_intro
   plot_cc_under_shift
   plot_method_comparison
   plot_distribution_matching
   plot_emq_convergence
   plot_neural_quantifiers
   plot_quanet
   plot_quantification_trees
   plot_readme
   plot_elm
   plot_protocols
   plot_error_by_shift
   plot_grid_search
   plot_confidence_regions
   plot_calibration
   plot_multiclass
   real_datasets_loading
   real_datasets_evaluation
   plot_synthetic_intro
   plot_synthetic_shift
   plot_synthetic_prevalence
   plot_synthetic_difficulty
   plot_synthetic_shift_types
   plot_synthetic_quantifiers
