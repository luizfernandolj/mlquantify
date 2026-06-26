.. _datasets:

.. _real_world_datasets:

.. currentmodule:: mlquantify.datasets

====================
Real-World Datasets
====================

Alongside the synthetic generator (:ref:`synthetic_datasets`), ``mlquantify.datasets``
ships **25 dataset fetchers** that download and cache well-known quantification
benchmarks. They follow the same conventions as :mod:`sklearn.datasets`: every
loader is a keyword-only ``fetch_<name>(...)`` function that returns a
:class:`Bunch` (or a plain ``(X, y)`` tuple), caching the raw file under a local
``data_home`` so it is downloaded only once.

What sets them apart from the scikit-learn loaders is the optional
**quantification protocol**: pass ``protocol="app"`` (or another protocol) and the
returned :class:`Bunch` additionally carries ``.samples`` and ``.prevalences`` —
a collection of test *bags* with known class prevalence, ready to score a
quantifier against.

.. code-block:: python

    from mlquantify.datasets import fetch_mushroom

    b = fetch_mushroom()                       # plain load -> Bunch(data, target, ...)
    X, y = fetch_mushroom(return_X_y=True)     # sklearn-style tuple

    # quantification mode: 1000 bags of 500 instances drawn with the
    # Artificial Prevalence Protocol
    b = fetch_mushroom(protocol="app", n_samples=1000, sample_size=500, random_state=0)
    b.samples        # list of index arrays into b.data, one per bag
    b.prevalences    # (n_samples, n_classes) array of TRUE prevalences

Shared configuration
====================================================================

Every fetcher accepts the same base keyword arguments:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Parameter
     - Meaning
   * - ``data_home=None``
     - Folder used to cache the downloaded file(s). Defaults to a ``_data/``
       directory next to the package.
   * - ``download_if_missing=True``
     - If ``False``, raise instead of downloading when the cache is empty.
   * - ``return_X_y=False``
     - Return ``(X, y)`` instead of a :class:`Bunch`.
   * - ``as_frame=False``
     - Return ``.data`` as a :class:`pandas.DataFrame` and ``.target`` as a
       :class:`pandas.Series` (with a combined ``.frame``), where applicable.
   * - ``n_retries=3``
     - Number of download attempts before giving up.
   * - ``delay=1.0``
     - Seconds to wait between attempts.
   * - ``protocol=None``
     - Quantification sampling protocol: ``None`` (no bags), ``"app"``, ``"npp"``,
       ``"upp"``, ``"ppp"``, or an mlquantify protocol instance. When set, the
       :class:`Bunch` also exposes ``.samples``, ``.prevalences`` and ``.protocol``.
   * - ``n_samples=1000``
     - Number of prevalence points (bags) the protocol generates.
   * - ``sample_size=500``
     - Instances per bag.
   * - ``random_state=None``
     - Seed forwarded to the protocol.

The ``protocol`` / ``n_samples`` / ``sample_size`` triple is the
quantification-specific part of the API; see :func:`make_protocol` and
:ref:`synthetic_datasets` for how protocols turn a labelled table into evaluation
bags.

Per-dataset configuration
====================================================================

Each dataset adds a few of its own keyword arguments on top of the shared ones
(shown in the *Fetcher (key params)* column). ``target_col=`` lets you override
the column used as the label for the tabular CSV loaders.

Binary (tabular)
--------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 36 20 18

   * - Dataset
     - Fetcher (key params)
     - Task / classes
     - Source
   * - Mushroom
     - ``fetch_mushroom(target_col=None)``
     - Binary (2) · easy
     - UCI #73
   * - Banknote Authentication
     - ``fetch_banknote_authentication(target_col=None)``
     - Binary (2) · easy
     - UCI #267
   * - Haberman's Survival
     - ``fetch_haberman_survival(target_col=None)``
     - Binary (2) · hard
     - UCI #43
   * - Pima Indians Diabetes
     - ``fetch_pima_diabetes(target_col=None)``
     - Binary (2) · hard
     - jbrownlee mirror
   * - MiniBooNE Particle ID
     - ``fetch_miniboone()``
     - Binary (2) · med-hard
     - UCI #199

Multiclass (tabular)
--------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 36 20 18

   * - Dataset
     - Fetcher (key params)
     - Task / classes
     - Source
   * - Optical / Pen-Based Digits
     - ``fetch_digits_optical_penbased(which='optical'|'penbased', target_col=None)``
     - Multiclass (10) · easy
     - UCI #80 / #81
   * - Dry Bean
     - ``fetch_dry_bean(target_col=None)``
     - Multiclass (7) · easy-med
     - UCI #602
   * - Covertype
     - ``fetch_covertype(target_col=None)``
     - Multiclass (7) · hard
     - UCI #31
   * - Yeast
     - ``fetch_yeast(target_col=None)``
     - Multiclass (10) · hard
     - UCI #110
   * - Sensorless Drive Diagnosis
     - ``fetch_sensorless_drive(target_col=None)``
     - Multiclass (11) · med-hard
     - UCI #325
   * - Statlog (Shuttle)
     - ``fetch_statlog_shuttle(target_col=None)``
     - Multiclass (7) · hard (imbalanced)
     - UCI #148

Ordinal
--------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 36 20 18

   * - Dataset
     - Fetcher (key params)
     - Task / classes
     - Source
   * - Wine Quality
     - ``fetch_wine_quality(target_col=None)``
     - Ordinal (quality 3–9)
     - UCI #186
   * - LeQua 2024
     - ``fetch_lequa2024(task='T1'|'T2'|'T3'|'T4', include_test=False)``
     - T1 binary · T2 multiclass-28 · T3 ordinal · T4 binary (covariate)
     - Zenodo

Text
--------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 36 20 18

   * - Dataset
     - Fetcher (key params)
     - Task / classes
     - Source
   * - RCV1-v2
     - ``fetch_rcv1_v2()``
     - Multilabel topic (sparse)
     - via scikit-learn
   * - 20 Newsgroups
     - ``fetch_newsgroups20(subset='train'|'test')``
     - Multiclass (20)
     - figshare
   * - IMDB
     - ``fetch_imdb(subset='train'|'test')``
     - Binary (2)
     - Stanford
   * - Multi-Domain Sentiment
     - ``fetch_multidomain_sentiment(domain='books'|'dvd'|'electronics'|'kitchen')``
     - Binary (2) · covariate / domain shift
     - JHU (Blitzer)

Temporal / time-shift
--------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 36 20 18

   * - Dataset
     - Fetcher (key params)
     - Task / classes
     - Source
   * - Sentiment140
     - ``fetch_sentiment140()``
     - Binary (2) · temporal
     - Stanford
   * - Electricity (Elec2)
     - ``fetch_electricity_elec2(target_col=None)``
     - Binary (2) · concept drift
     - scikit-multiflow
   * - Airlines (flight delay)
     - ``fetch_airlines(target_col=None)``
     - Binary (2) · concept drift
     - scikit-multiflow
   * - Online News Popularity
     - ``fetch_online_news_popularity(threshold=1400)``
     - Binary (popularity thresholded at ``threshold``)
     - UCI #332

Image, graph & synthetic (other shift)
--------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 36 20 18

   * - Dataset
     - Fetcher (key params)
     - Task / classes
     - Source
   * - SEA Concepts
     - ``fetch_sea_concepts(n=50000, drift=True, noise=0.1)``
     - Binary (2) · concept drift (generated, no download)
     - synthetic
   * - MNIST → USPS
     - ``fetch_mnist_usps(domain='mnist'|'usps', subset='train'|'test'|'all')``
     - Multiclass (10) · covariate shift
     - LIBSVM
   * - CIFAR-10
     - ``fetch_cifar10(subset='train'|'test'|'all')``
     - Multiclass (10) · label / prior shift
     - Toronto
   * - Planetoid (Cora / CiteSeer / PubMed)
     - ``fetch_planetoid_cora_citeseer_pubmed(name='cora'|'citeseer'|'pubmed')``
     - Multiclass (3–7) · graph, covariate / structural shift
     - GitHub (Planetoid)

Optional download progress
====================================================================

The fetchers print nothing by default and pull in **no** progress dependency. To
display download progress, register any callable with :func:`set_progress_hook`;
every fetcher will then report its downloads to it as ``hook(downloaded, total, url)``
(``total`` is ``None`` when the server sends no ``Content-Length``). This keeps the
choice of progress tool — plain ``print``, ``logging`` or ``tqdm`` — entirely yours:

.. code-block:: python

    from mlquantify.datasets import set_progress_hook
    from tqdm import tqdm          # your dependency, not mlquantify's

    bars = {}
    def show(downloaded, total, url):
        bar = bars.setdefault(url, tqdm(total=total, unit="B", unit_scale=True))
        bar.update(downloaded - bar.n)

    set_progress_hook(show)        # all fetchers now report progress

References
====================================================================

The table below lists, for each dataset, the quantification works in the
literature that use it (numbers refer to the reference list underneath).

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Dataset
     - Used by — quantification papers
   * - Mushroom
     - [1]_ [2]_ [3]_
   * - Banknote Authentication
     - — (new addition)
   * - Haberman's Survival
     - [1]_ [4]_ [5]_ [6]_
   * - Pima Indians Diabetes
     - [1]_ [5]_ [6]_ [7]_ [8]_
   * - MiniBooNE Particle ID
     - [9]_
   * - Optical / Pen-Based Digits
     - [10]_ [11]_
   * - Dry Bean
     - — (new addition)
   * - Covertype
     - [2]_ [6]_ [10]_ [12]_ [13]_
   * - Yeast
     - [5]_ [6]_ [7]_ [14]_
   * - Sensorless Drive Diagnosis
     - — (new addition)
   * - Statlog (Shuttle)
     - [8]_ [9]_
   * - Wine Quality
     - [2]_ [6]_ [7]_ [11]_ [14]_ [15]_
   * - LeQua 2024 (T1–T4)
     - [16]_ [17]_
   * - RCV1-v2
     - [18]_ [19]_
   * - 20 Newsgroups
     - [19]_
   * - IMDB
     - [19]_ [20]_
   * - Multi-Domain Sentiment
     - — (new addition)
   * - Sentiment140
     - [21]_
   * - Electricity (Elec2)
     - — (new addition)
   * - Airlines
     - — (new addition)
   * - Online News Popularity
     - [21]_
   * - SEA Concepts
     - — (new addition)
   * - MNIST → USPS
     - [22]_
   * - CIFAR-10
     - [22]_ [23]_
   * - Planetoid (Cora / CiteSeer / PubMed)
     - [24]_

Datasets marked *new addition* are recent additions to the library that are not
yet used by a quantification work in the surveyed bibliography; see the *Source*
column above for each dataset's original reference.

.. [1] Bella, A., Ferri, C., Hernández-Orallo, J., Ramírez-Quintana, M.J. (2010).
   Quantification via Probability Estimators. *IEEE ICDM 2010*, 737–742.
.. [2] Maletzke, A., dos Reis, D., Cherman, E., Batista, G. (2019).
   DyS: A Framework for Mixture Models in Quantification. *AAAI 2019*, 4552–4560.
.. [3] Maletzke, A. (2019). Binary Quantification in Non-Stationary Scenarios.
   PhD thesis, ICMC, Universidade de São Paulo.
.. [4] Barranquero, J., González, P., Díez, J., del Coz, J.J. (2013). On the study
   of nearest neighbor algorithms for prevalence estimation in binary problems.
   *Pattern Recognition* 46(2), 472–482.
.. [5] Pérez-Gállego, P., Castaño, A., Quevedo, J.R., del Coz, J.J. (2019).
   Dynamic ensemble selection for quantification tasks. *Information Fusion* 45, 1–15.
.. [6] Bunse, M., et al. (eds.) (2024; 2025). Proceedings of the 4th / 5th
   International Workshop on Learning to Quantify (LQ 2024 / LQ 2025).
.. [7] González-Castro, V., Alaiz-Rodríguez, R., Alegre, E. (2013). Class
   distribution estimation based on the Hellinger distance. *Information Sciences*
   218, 146–164.
.. [8] Iyer, A., Nath, S., Sarawagi, S. (2014). Maximum Mean Discrepancy for Class
   Ratio Estimation: Convergence Bounds and Kernel Selection. *ICML 2014*.
.. [9] Schumacher, T., Strohmaier, M., Lemmerich, F. (2023). A Comparative
   Evaluation of Quantification Methods. arXiv preprint.
.. [10] Joachims, T. (2005). A Support Vector Method for Multivariate Performance
   Measures. *ICML 2005*, 377–384.
.. [11] dos Reis, D., Maletzke, A., Cherman, E., Batista, G. (2018). One-class
   Quantification. *ECML-PKDD 2018*, 273–289.
.. [12] Maletzke, A., dos Reis, D., Hassan, W., Batista, G. (2021). Accurately
   Quantifying under Score Variability (MoSS/DySyn). *IEEE ICDM 2021*.
.. [13] Xue, J.C., Weiss, G.M. (2009). Quantification and semi-supervised
   classification methods for handling changes in class distribution.
   *ACM SIGKDD 2009*, 897–906.
.. [14] Moreo, A., Salvati, M. (2025). An Efficient Method for Deriving Confidence
   Intervals in Aggregative Quantification.
.. [15] Hassan, W., Maletzke, A., Batista, G. (2020). Accurately Quantifying a
   Billion Instances per Second. *IEEE DSAA 2020*.
.. [16] Pérez-Mon, O., Moreo, A., del Coz, J.J., González, P. (2024).
   Quantification using permutation-invariant networks based on histograms
   (HistNetQ). *Machine Learning*.
.. [17] Pérez-Mon, O., del Coz, J.J., González, P. (2025). Quantification via
   Gaussian Latent Space Representations (GMNet).
.. [18] Hassan, W., Maletzke, A., Batista, G. (2021). Pitfalls in Quantification
   Assessment.
.. [19] Esuli, A., Fabris, A., Moreo, A., Sebastiani, F. (2023). Learning to
   Quantify. Springer, The Information Retrieval Series.
.. [20] Esuli, A., Moreo, A., Sebastiani, F. (2018). A Recurrent Neural Network for
   Sentiment Quantification (QuaNet). *ACM CIKM 2018*, 1775–1778.
.. [21] Li, et al. (2025). Quantification over Time. arXiv preprint.
.. [22] Alexandari, A., Kundaje, A., Shrikumar, A. (2020). Maximum Likelihood with
   Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation. *ICML 2020*.
.. [23] Firat, A. (2016). Unified Framework for Quantification. arXiv:1606.00868.
.. [24] Damke, C., Hüllermeier, E. (2025). Adjusted Count Quantification Learning on
   Graphs. preprint.

.. seealso::

   - :ref:`synthetic_datasets` — the :func:`make_quantification` generator for
     fully controlled prior / covariate / concept shift.
   - :mod:`mlquantify.datasets` — the full API reference for every fetcher.
