.. _evaluation:

Metrics and scoring: quantifing the performance of a quantifier
===============================================================

.. currentmodule:: mlquantify.evaluation

The measures implemented in this module are shown in the book *Learning to Quantifier*, and can be listed as follows:

.. list-table::
    :header-rows: 1

    * - measure
      - abbreviation
      - First proposed for quantification
      - Return type
    * - `Absolute error <generated/mlquantify.evaluation.measures.absolute_error.html>`_
      - ae
      - --
      - array-like
    * - `Mean Absolute Error <generated/mlquantify.evaluation.measures.mean_absolute_error.html>`_
      - mae
      - `Saerens et al. (2002) <https://ieeexplore.ieee.org/abstract/document/6789744>`_
      - float
    * - `Normalized Absolute Error <generated/mlquantify.evaluation.measures.normalized_absolute_error.html>`_
      - nae
      - `Esuli and Sebastiani (2014) <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f0401ab7579c94dca7fdc5fba6d8f1665d155a58#page=8>`_
      - float
    * - `Relative Absolute Error <generated/mlquantify.evaluation.measures.relative_absolute_error.html>`_
      - rae
      - `González-Castro et al. (2010) <https://link.springer.com/chapter/10.1007/978-3-642-13022-9_29>`_
      - float
    * - `Normalized Relative Absolute Error <generated/mlquantify.evaluation.measures.normalized_relative_absolute_error.html>`_
      - nrae
      - `Esuli and Sebastiani (2014) <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f0401ab7579c94dca7fdc5fba6d8f1665d155a58#page=8>`_
      - float
    * - `Squared Error <generated/mlquantify.evaluation.measures.squared_error.html>`_
      - se
      - `Bella et al. (2011) <https://ieeexplore.ieee.org/abstract/document/5694031>`_
      - float
    * - `Mean Squared Error <generated/mlquantify.evaluation.measures.mean_squared_error.html>`_
      - mse
      - --
      - float
    * - `Kullback Leibler Divergence <generated/mlquantify.evaluation.measures.kullback_leibler_divergence.html>`_
      - kld
      - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
      - array-like
    * - `Normalized Kullback Leibler Divergence <generated/mlquantify.evaluation.measures.normalized_kullback_leibler_divergence.html>`_
      - nkld
      - `Esuli and Sebastiani (2014) <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f0401ab7579c94dca7fdc5fba6d8f1665d155a58#page=8>`_
      - float
  
.. note::
    **Return type**
    
    When inserting the measure in the `scoring` parameter of :class:`~mlquantify.model_selection.GridSearchQ` or :class:`~mlquantify.evaluation.protocol.APP` classes, the name of the measure should be passed with its acronym, for example "mae" for :func:`~mlquantify.evaluation.measures.mean_absolute_error`.