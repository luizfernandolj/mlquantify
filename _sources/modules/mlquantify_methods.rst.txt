MLQuantify Methods
==================

The table below lists all the quantification methods available in the ``mlquantify`` library, their references, multiclass support, and type (aggregative, meta, or non-aggregative).

.. note::
   In ``binary`` classification problems, methods that do not natively support multiclass classification (marked ``No`` in the ``Multiclass`` column) remain applicable through standard reduction strategies like **one-vs-rest** or **one-vs-one**.

.. list-table::
   :widths: 25 35 20 20 20
   :header-rows: 1

   * - Method
     - Reference
     - Multiclass
     - Type
     - Module
   * - :class:`~mlquantify.adjust_counting.CC`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.PCC`
     - `Bella et al. (2010) <https://ieeexplore.ieee.org/document/5694031>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.AC`
     - `Firat (2016) <https://arxiv.org/abs/1606.00868>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.PAC`
     - `Firat (2016) <https://arxiv.org/abs/1606.00868>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.TAC`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.TX`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.TMAX`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.T50`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.MS`
     - `Forman (2006) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.MS2`
     - `Forman (2006) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.FM`
     - `Friedman et al. (2015) <https://jerryfriedman.su.domains/talks/qc.pdf>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.adjust_counting.CDE`
     - `Xue & Weiss (2009) <https://dl.acm.org/doi/abs/10.1145/1557019.1557117>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.adjust_counting`
   * - :class:`~mlquantify.likelihood.EMQ`
     - `Saerens et al. (2002) <https://ieeexplore.ieee.org/abstract/document/6789744>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.likelihood`
   * - :class:`~mlquantify.mixture.DyS`
     - `Maletzke et al. (2019) <https://ojs.aaai.org/index.php/AAAI/article/view/4376>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.mixture`
   * - :class:`~mlquantify.mixture.HDy`
     - `Gonzalez et al. (2012) <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.mixture`
   * - :class:`~mlquantify.mixture.SMM`
     - `Hassan et al. (2020) <https://ieeexplore.ieee.org/abstract/document/9260028>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.mixture`
   * - :class:`~mlquantify.mixture.SORD`
     - `Maletzke et al. (2019) <https://ojs.aaai.org/index.php/AAAI/article/view/4376>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.mixture`
   * - :class:`~mlquantify.mixture.HDx`
     - `Gonzalez et al. (2012) <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_
     - No
     - Non-aggregative
     - :mod:`~mlquantify.mixture`
   * - :class:`~mlquantify.mixture.MMD_RKHS`
     - `Iyer et al. (2014) <https://proceedings.mlr.press/v32/iyer14.html>`_
     - No
     - Non-aggregative
     - :mod:`~mlquantify.mixture`
   * - :class:`~mlquantify.neighbors.KDEyML`
     - `Moreo et al. (2025) <https://link.springer.com/article/10.1007/s10994-024-06726-5>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.neighbors`
   * - :class:`~mlquantify.neighbors.KDEyHD`
     - `Moreo et al. (2025) <https://link.springer.com/article/10.1007/s10994-024-06726-5>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.neighbors`
   * - :class:`~mlquantify.neighbors.KDEyCS`
     - `Moreo et al. (2025) <https://link.springer.com/article/10.1007/s10994-024-06726-5>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.neighbors`
   * - :class:`~mlquantify.neighbors.PWK`
     - `Barraquero et al. (2013) <https://www.sciencedirect.com/science/article/pii/S0031320312003391>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.neighbors`
   * - :class:`~mlquantify.meta.EnsembleQ`
     - `Pérez-Gállego et al. (2017) <https://www.sciencedirect.com/science/article/pii/S1566253516300628>`_ and `Pérez-Gállego et al. (2019) <https://www.sciencedirect.com/science/article/pii/S1566253517303652>`_
     - Method dependent
     - Meta
     - :mod:`~mlquantify.meta`
   * - :class:`~mlquantify.meta.QuaDapt`
     - `Ortega et al. (2025) <https://hal.science/hal-04942724/document>`_
     - Method dependent
     - Meta
     - :mod:`~mlquantify.meta`
   * - :class:`~mlquantify.meta.AggregativeBootstrap`
     - `Moreo & Salvati (2025) <https://iris.cnr.it/bitstream/20.500.14243/555966/1/BootsCI.LQ2025.pdf>`_
     - Method dependent
     - Meta
     - :mod:`~mlquantify.meta`
