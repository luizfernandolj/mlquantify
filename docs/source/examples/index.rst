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


.. toctree::
   :hidden:

   plot_quantification_intro
   plot_cc_under_shift
   plot_method_comparison
   plot_distribution_matching
   plot_emq_convergence
   plot_protocols
   plot_error_by_shift
   plot_grid_search
   plot_confidence_regions
   plot_multiclass
