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
   * - :class:`~mlquantify.counting.CC`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.PCC`
     - `Bella et al. (2010) <https://ieeexplore.ieee.org/document/5694031>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.GACC`
     - `Firat (2016) <https://arxiv.org/abs/1606.00868>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.GPACC`
     - `Firat (2016) <https://arxiv.org/abs/1606.00868>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.TAC`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.TX`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.TMAX`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.T50`
     - `Forman (2005) <https://link.springer.com/chapter/10.1007/11564096_55>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.MS`
     - `Forman (2006) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.MS2`
     - `Forman (2006) <https://link.springer.com/article/10.1007/s10618-008-0097-y>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.counting.FM`
     - `Friedman et al. (2015) <https://jerryfriedman.su.domains/talks/qc.pdf>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.counting`
   * - :class:`~mlquantify.likelihood.CDE`
     - `Xue & Weiss (2009) <https://dl.acm.org/doi/abs/10.1145/1557019.1557117>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.likelihood`
   * - :class:`~mlquantify.likelihood.MLPE`
     - `Saerens et al. (2002) <https://ieeexplore.ieee.org/abstract/document/6789744>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.likelihood`
   * - :class:`~mlquantify.likelihood.EMQ`
     - `Saerens et al. (2002) <https://ieeexplore.ieee.org/abstract/document/6789744>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.likelihood`
   * - :class:`~mlquantify.matching.DyS`
     - `Maletzke et al. (2019) <https://ojs.aaai.org/index.php/AAAI/article/view/4376>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.matching.HDy`
     - `Gonzalez et al. (2012) <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.matching.SMM`
     - `Hassan et al. (2020) <https://ieeexplore.ieee.org/abstract/document/9260028>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.matching.SORD`
     - `Maletzke et al. (2019) <https://ojs.aaai.org/index.php/AAAI/article/view/4376>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.matching.HDx`
     - `Gonzalez et al. (2012) <https://www.sciencedirect.com/science/article/pii/S0020025512004069>`_
     - No
     - Non-aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.matching.MMD_RKHS`
     - `Iyer et al. (2014) <https://proceedings.mlr.press/v32/iyer14.html>`_
     - No
     - Non-aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.matching.KDEyML`
     - `Moreo et al. (2025) <https://link.springer.com/article/10.1007/s10994-024-06726-5>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.matching.KDEyHD`
     - `Moreo et al. (2025) <https://link.springer.com/article/10.1007/s10994-024-06726-5>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.matching.KDEyCS`
     - `Moreo et al. (2025) <https://link.springer.com/article/10.1007/s10994-024-06726-5>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.matching`
   * - :class:`~mlquantify.neighbors.PWK`
     - `Barraquero et al. (2013) <https://www.sciencedirect.com/science/article/pii/S0031320312003391>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.neighbors`
   * - :class:`~mlquantify.elm.SVMQ`
     - `Barranquero et al. (2015) <https://www.sciencedirect.com/science/article/pii/S0031320314003501>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.elm`
   * - :class:`~mlquantify.elm.SVMKLD`
     - `Esuli & Sebastiani (2015) <https://dl.acm.org/doi/10.1145/2700406>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.elm`
   * - :class:`~mlquantify.elm.SVMNKLD`
     - `Esuli & Sebastiani (2015) <https://dl.acm.org/doi/10.1145/2700406>`_
     - No
     - Aggregative
     - :mod:`~mlquantify.elm`
   * - :class:`~mlquantify.readme.ReadMe`
     - `Hopkins & King (2010) <https://onlinelibrary.wiley.com/doi/10.1111/j.1540-5907.2009.00428.x>`_
     - Yes
     - Non-aggregative
     - :mod:`~mlquantify.readme`
   * - :class:`~mlquantify.readme.ReadMe2`
     - `Jerzak et al. (2022) <https://www.cambridge.org/core/journals/political-analysis/article/improved-method-of-automated-nonparametric-content-analysis-for-social-science/60343978B77598E1E4229E7B85CD2081>`_
     - Yes
     - Non-aggregative
     - :mod:`~mlquantify.readme`
   * - :class:`~mlquantify.tree.QuantificationTree`
     - `Milli et al. (2013) <https://ieeexplore.ieee.org/document/6729537>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.tree`
   * - :class:`~mlquantify.tree.QuantificationForest`
     - `Milli et al. (2013) <https://ieeexplore.ieee.org/document/6729537>`_
     - Yes
     - Aggregative
     - :mod:`~mlquantify.tree`
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
   * - :class:`~mlquantify.neural.QuaNet`
     - `Esuli et al. (2018) <https://doi.org/10.1016/j.patrec.2019.11.012>`_
     - Yes
     - Neural
     - :mod:`~mlquantify.neural`
   * - :class:`~mlquantify.neural.HistNetQ`
     - `Pérez-Mon et al. (2024) <https://doi.org/10.1007/s00521-024-10721-1>`_
     - Yes
     - Neural
     - :mod:`~mlquantify.neural`
   * - :class:`~mlquantify.neural.GMNet`
     - `Pérez-Mon et al. (2025) <https://arxiv.org/abs/2501.13638>`_
     - Yes
     - Neural
     - :mod:`~mlquantify.neural`
