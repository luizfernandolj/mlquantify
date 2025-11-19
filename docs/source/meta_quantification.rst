.. _meta_quantification:

Meta-Quantification
--------------------

Meta-quantification methods combine multiple quantification approaches to improve the accuracy and robustness of class prevalence estimates. These methods leverage the strengths of different quantification techniques, often by integrating their outputs or using ensemble strategies to mitigate individual method weaknesses. Meta-quantification can involve techniques such as stacking, voting, or weighted averaging of predictions from various quantifiers.

The main meta-quantification methods will be defined next, with details about their specific approaches and differences.

.. toctree::
   :maxdepth: 2

   modules/ensemble.rst
   modules/bootstrap.rst
   modules/scores_adaptation.rst