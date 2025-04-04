.. _protocol:

========
Protocol
========

.. currentmodule:: mlquantify.evaluation.protocol

Protocols are a set of methods that are used to evaluate the performance of quantifiers. They are designed to provide a standardized way of measuring the performance of different quantifiers, and to allow for easy comparison between them. Two main protocols were defined in the literature: the :class:`~mlquantify.evaluation.protocol.APP` (Artificial Prevalence Protocol) and the :class:`~mlquantify.evaluation.protocol.NPP` (Natural Prevalence Protocol).

The base parameters for any protocol are:

.. list-table:: Base parameters for any protocol
    :header-rows: 1

    * - Parameter
      - Type
    * - models
      - Union[List[Union[str, Quantifier]], str, Quantifier]
    * - learner
      - BaseEstimator = None
    * - n_jobs
      - int = 1
    * - random_state
      - int = 32
    * - verbose
      - bool = False
    * - return_type
      - str = "predictions" (can be "table")
    * - measures
      - List[str] = None
    * - columns
      - ["ITERATION", "QUANTIFIER", "REAL_PREVS", "PRED_PREVS", "BATCH_SIZE"]

The `return_type` parameter can be set to "predictions" or "table". If set to "predictions", the protocol will return 3 array-like objects: the name of the quantifiers, the real prevalences and the predicted prevalences. If set to "table", the protocol will return a pandas DataFrame with the results of the evaluation, containing:

.. list-table:: Evaluation output parameters for "table" return_type
    :header-rows: 1

    * - Parameter
      - Type
      - Description
    * - ITERATION
      - int
      - Iteration number
    * - QUANTIFIER
      - str
      - Name of the quantifier
    * - REAL_PREVS
      - array-like 
      - Real prevalences of the batch, with size (n_classes)
    * - PRED_PREVS
      - array-like
      - Pred prevalences of the batch, with size (n_classes)
    * - BATCH_SIZE
      - int
      - Size of the batch
    * - measure\ :sub:`i`
      - float, array-like
      - Value of the measure\ :sub:`i` to be applied to (REAL_PREVS\ :sub:`i`, PRED_PREVS\ :sub:`i`)

  

For the `models`, the examples of usage can be found in :ref:`app_general` and :ref:`app_selected` sections.

The `columns` parameter can be a list of column names if you want to change the default columns of the output table.

.. note::

    If you want to make your own protocol, see :ref:`building_a_protocol` for more details.