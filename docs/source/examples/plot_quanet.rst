.. _sphx_quanet:

=======================================
QuaNet: deep quantification with LSTMs
=======================================

:class:`~mlquantify.neural.QuaNet` (Esuli, Moreo & Sebastiani, 2018) is an
*asymmetric* neural quantifier: it keeps a regular probabilistic classifier
and trains a recurrent network **on top of it**. For each bag, the documents'
embeddings and posterior probabilities are sorted and fed through a
bidirectional LSTM; the resulting *quantification embedding* is concatenated
with the estimates of several classical quantifiers (CC, ACC, PCC, ...) and a
feed-forward head outputs the corrected prevalence vector. The network is
trained on bags sampled across the prevalence simplex, so it learns how the
classifier's errors distort counts — and how to undo them.

QuaNet needs two ingredients: an ``estimator`` with ``predict_proba`` and an
embedding source with ``transform`` (either the estimator itself or a separate
``embedder`` — here a PCA; for text, a ``TfidfVectorizer`` or a
:class:`~mlquantify.neural.TorchClassifierWrapper` around your own network).

.. note::

   Requires PyTorch (``pip install torch``). The settings below are
   deliberately small so the example runs quickly; for real problems use more
   epochs, larger samples, and GPU (``device="cuda"``).

.. plot::

    import os
    import tempfile

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.neural import QuaNet
    from mlquantify.visualization import DiagonalDisplay

    set_config(prevalence_return_type="array")

    # A balanced training sample plus test bags on a grid of prevalence
    # values from 0 to 1.
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        batch_size=100, return_train=True, train_size=2000,
        train_prevalence=[0.5, 0.5],
        prevalence="grid", n_prevalences=11, repeats=2,
        n_features=10, n_informative=6, class_sep=1.5, random_state=0,
    )
    Xtr = Xtr.astype(np.float32)

    quanet = QuaNet(
        estimator=LogisticRegression(max_iter=1000),
        embedder=PCA(n_components=8),
        sample_size=64,
        n_epochs=10, tr_iter=100, va_iter=20, patience=3,
        lstm_hidden_size=32, ff_layers=(256, 128),
        checkpointdir=os.path.join(tempfile.gettempdir(), "quanet_example"),
        device="cpu", random_state=0,
    )
    quanet.fit(Xtr, ytr)

    pred = np.vstack([quanet.predict(Xb.astype(np.float32)) for Xb in Xs])

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    DiagonalDisplay.from_predictions(
        prevs, pred, ax=ax, color="#2a9d8f", alpha=0.5, s=22,
    )
    mae = float(np.mean(np.abs(pred - prevs)))
    ax.set_title(f"QuaNet  (MAE = {mae:.3f})")
    fig.tight_layout()

Even with this deliberately tiny configuration the estimates line up with the
:math:`y = x` diagonal across the whole prevalence range: the LSTM has learned
the classifier's bias pattern from the training bags and corrects it on the
shifted test bags.

.. seealso::

   - :ref:`neural_quantifiers` — QuaNet's architecture and every
     hyper-parameter, plus the symmetric quantifiers HistNetQ and GMNet.
   - :ref:`sphx_neural_quantifiers` — the symmetric neural quantifiers on the
     same kind of diagonal plot.
