.. _non_aggregative_quantification:

Non-Aggregative Quantification
------------------------------


Non-aggregative quantification methods estimate class prevalence directly from the data without relying on aggregating individual predictions. These methods often involve specialized algorithms that are designed to infer the overall distribution of classes in a dataset based on certain characteristics or patterns observed in the data.
Unlike aggregative methods, non-aggregative approaches do not follow the traditional aggregate step after making individual predictions. Instead, they may utilize techniques such as distribution matching, density estimation, or other statistical methods to directly estimate class prevalences.

The main non-aggregative quantification methods will be defined next, with details about their specific approaches and differences.

.. toctree::
   :maxdepth: 2

   modules/mixture_models_non_agg.rst