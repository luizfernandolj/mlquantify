.. _neural_quantifiers:

.. currentmodule:: mlquantify.neural

==================
Neural Quantifiers
==================

Neural quantifiers learn a direct mapping from a *bag of instances* to a
prevalence vector, without relying on a hand-crafted aggregation formula.
They are trained end-to-end to minimise a quantification loss and can
exploit deep feature representations that are inaccessible to analytical
methods.

.. admonition:: PyTorch required

   Neural quantifiers depend on ``torch``. Install it with::

      pip install torch

.. contents:: Contents
   :local:
   :depth: 2

----

QuaNet — Quantification Network
=================================

:class:`QuaNet` (Esuli et al., 2018) is a recurrent neural network that reads
a **set** of instance embeddings produced by a base classifier and predicts
the prevalence vector for that set.

**Architecture:**

1. The base classifier (``estimator``) produces a fixed-size embedding for
   each test instance (via its ``transform`` or ``predict_proba`` output).
2. An LSTM reads the sequence of embeddings (in random order) to produce a
   context vector summarising the set.
3. The context vector is concatenated with *auxiliary quantification statistics*
   (CC, PCC, and ACC estimates) computed on the current batch.
4. A feed-forward head maps the concatenated vector to a prevalence vector
   with a softmax output.

**Why it exists:** QuaNet learns to exploit patterns in instance embeddings
that rule-based aggregation methods cannot capture. On large text datasets
where embeddings carry rich distributional information, it has shown
competitive or superior performance to DyS and EMQ.

Parameters
----------

.. list-table::
   :widths: 22 15 63
   :header-rows: 1

   * - Parameter
     - Default
     - Explanation
   * - ``estimator``
     - required
     - A classifier that (a) produces posterior probabilities via
       ``predict_proba`` and (b) optionally exposes a ``transform`` method for
       dense embeddings. The predictions are used as LSTM inputs.
   * - ``device``
     - ``'cpu'``
     - PyTorch device. Set to ``'cuda'`` to use a GPU if available. Training
       is significantly faster on GPU for large datasets.
   * - ``hidden_size``
     - ``64``
     - Size of the LSTM hidden state. Larger values give more capacity but
       require more data. Try 32, 64, 128 depending on dataset size.
   * - ``n_hidden_layers``
     - ``1``
     - Number of LSTM layers. More layers capture longer-range dependencies in
       the embedding sequence but are slower to train.
   * - ``lstm_hidden_size``
     - ``32``
     - Hidden size per LSTM layer.
   * - ``drop_p``
     - ``0.5``
     - Dropout probability in the feed-forward head. Reduce to 0.2–0.3 if
       training data is large; increase to 0.6–0.7 to combat overfitting on
       small datasets.
   * - ``batch_size``
     - ``64``
     - Number of instances per training mini-batch. Larger batches are faster
       on GPU; smaller batches provide more gradient-update steps per epoch.
   * - ``max_epoch``
     - ``100``
     - Maximum training epochs. Early stopping kicks in if validation loss
       stops improving.
   * - ``patience``
     - ``10``
     - Early-stopping patience (epochs without improvement before stopping).
   * - ``lr``
     - ``1e-3``
     - Adam learning rate. Reduce to ``1e-4`` if training is unstable.
   * - ``val_split``
     - ``0.3``
     - Fraction of training data held out as validation (for early stopping).

Examples
--------

.. code-block:: python

   # Requires PyTorch
   from mlquantify.neural import QuaNet
   from sklearn.linear_model import LogisticRegression
   from sklearn.datasets import make_classification
   from sklearn.model_selection import train_test_split

   X, y = make_classification(n_samples=2000, n_features=20,
                              weights=[0.7, 0.3], random_state=42)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42)

   # QuaNet uses the classifier's predict_proba as embedding
   q = QuaNet(
       estimator=LogisticRegression(),
       device='cpu',
       hidden_size=64,
       max_epoch=50,
       patience=5,
   )
   q.fit(X_train, y_train)
   print(q.predict(X_test))

.. note::

   QuaNet requires the estimator to be **pre-fitted** before ``QuaNet.fit``
   if you pass ``estimator_fitted=True``, or it will fit the estimator
   internally as part of the training pipeline.

When to Use QuaNet
-------------------

- **Large text datasets** where the base classifier produces rich embeddings
  (e.g. transformer-based models with ``transform``).
- **When EMQ / DyS plateau** and you have enough data and computation to
  train end-to-end.
- **Not recommended** for small datasets (< 1,000 instances) or when
  computation is constrained — analytical methods (EMQ, DyS) will be faster
  and likely more accurate.

.. seealso::

   :ref:`likelihood` for EMQ, which is faster and often competitive.
   :ref:`distribution_matching` for DyS / KDEyML.

