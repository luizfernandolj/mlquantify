.. _aggregative:

==========================
Aggregative Quantification
==========================

Aggregative quantifiers are a class of quantification methods that aggregates the results of a mid task, such as classification, i.e., the quantifier uses the predicted values (labels or scores) of the classifier to estimate the class distribution of the test set.
This types of quantifiers can be separated into two main groups: the mixture models and threshold methods, but there are also other methods that do not fit into these categories, such as:

.. list-table:: Other Aggregative Quantifiers
    :header-rows: 1

    * - quantifier
      - class
      - reference
    * - Classify and Count
      - `CC <generated/mlquantify.methods.aggregative.CC.html>`_
      - `Forman (2005) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
    * - Expectation Maximisation for Quantification
      - `EMQ <generated/mlquantify.methods.aggregative.EMQ.html>`_
      - `Saerens et al. (2002) <https://www.sciencedirect.com/science/article/abs/pii/S0020025512004069?casa_token=W6UksOigmp4AAAAA:ap8FK5mtpAzG-s8k2ygfRVgdIBYDGWjEi70ueJ546coP9F-VNaCKE5W_gsAv0bWQiwzt2QoAuLjP>`_
    * - Probabilistic Classify and Count
      - `PCC <generated/mlquantify.methods.aggregative.PCC.html>`_
      - `Bella et al. (2010) <https://ieeexplore.ieee.org/abstract/document/5694031>`_
    * - Friedman Method
      - `FM <generated/mlquantify.methods.aggregative.FM.html>`_
      - `Friedman <https://jerryfriedman.su.domains/talks/qc.pdf>`_
    * - Generalized Adjust Count
      - `GAC <generated/mlquantify.methods.aggregative.GAC.html>`_
      - `Firat (2008) <https://arxiv.org/abs/1606.00868>`_
    * - Generalized Probabilistic Adjusted Count
      - `GPAC <generated/mlquantify.methods.aggregative.GPAC.html>`_
      - `Firat (2008) <https://arxiv.org/abs/1606.00868>`_
    * - Nearest-Neighbor based Quantification
      - `PWK <generated/mlquantify.methods.aggregative.PWK.html>`_
      - `Barraquero et al. (2013) <https://www.sciencedirect.com/science/article/abs/pii/S0031320312003391?casa_token=YUdybEKGhhAAAAAA:HMhq6SBW_-c2FvOZtUE4qB0FRWbp_XnxjtkCBfQZU0aKO325EFP48uLLwfLWBzoSY9T6zdP0kSDk>`_

An important note is that all these listed methods are **multiclass** quantifiers, but the mixture models and threshold methods are **binary** quantifiers.


.. include:: mixture_models.rst
.. include:: threshold_methods.rst
.. include:: emq.rst


   