.. _sphx_real_datasets_loading:

===========================
Loading real-world datasets
===========================

``mlquantify.datasets`` ships 25 *fetchers* that download and cache well-known
quantification benchmarks — tabular, text, image, graph, a concept-drift stream,
and the LeQua 2024 competition. They follow the :mod:`sklearn.datasets`
conventions: every loader is a keyword-only ``fetch_<name>(...)`` that returns a
:class:`~mlquantify.datasets.Bunch` and caches the raw file so it is downloaded
only once.

Basic load
==========

.. code-block:: python

    from mlquantify.datasets import fetch_mushroom

    data = fetch_mushroom()      # downloads + caches on first call
    data.data                    # feature matrix, shape (n_samples, n_features)
    data.target                  # integer class labels
    data.target_names            # e.g. ['edible', 'poisonous']

scikit-learn-style outputs
==========================

.. code-block:: python

    # A plain (X, y) tuple instead of a Bunch:
    X, y = fetch_mushroom(return_X_y=True)

    # pandas DataFrame / Series, with a combined .frame:
    bunch = fetch_mushroom(as_frame=True)
    bunch.frame.head()

Caching and offline use
=======================

Every fetcher accepts the same caching controls:

.. code-block:: python

    # Where the raw files are cached (default: a _data/ dir next to the package).
    data = fetch_mushroom(data_home="~/mlquantify_data")

    # Fail instead of downloading when the cache is empty (e.g. offline / CI):
    data = fetch_mushroom(download_if_missing=False)

Other datasets follow the same pattern — e.g.
:func:`~mlquantify.datasets.fetch_dry_bean` (multiclass tabular),
:func:`~mlquantify.datasets.fetch_imdb` (text) or
:func:`~mlquantify.datasets.fetch_cifar10` (image). See the
:ref:`real_world_datasets` guide for the full catalogue and each dataset's extra
options.

.. seealso::

   - :ref:`real_world_datasets` — the full list of fetchers and their options.
   - :ref:`sphx_real_datasets_evaluation` — score a quantifier on a fetched
     dataset.
