.. _sphx_neural_quantifiers:

==========================================
Neural quantifiers: HistNetQ and GMNet
==========================================

The *symmetric* neural quantifiers :class:`~mlquantify.neural.HistNetQ` and
:class:`~mlquantify.neural.GMNet` learn to map a **bag of instances** straight to
a prevalence vector. You supply a small PyTorch *feature extractor* (FEM); the
library wraps it with a permutation-invariant *bag representation* (a
differentiable histogram for HistNetQ, a Gaussian mixture for GMNet) and a
softmax *quantification head*, then trains the whole thing end-to-end on bags
drawn across the prevalence simplex.

This example trains both on the same synthetic data and reads off their bias and
variance from true-vs-predicted diagonal plots, exactly as in
:ref:`sphx_method_comparison`.

.. note::

   These methods require PyTorch (``pip install torch``). The settings below are
   deliberately small so the example runs quickly; for real problems use larger
   bags and more training bags per epoch.

.. plot::

    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt

    from mlquantify import set_config
    from mlquantify.datasets import make_quantification
    from mlquantify.neural import HistNetQ, GMNet
    from mlquantify.visualization import DiagonalDisplay

    set_config(prevalence_return_type="array")

    # A balanced training sample plus test bags on a grid of prevalence
    # values across the whole [0, 1] range.
    Xtr, ytr, Xs, ys, prevs = make_quantification(
        batch_size=150, return_train=True, train_size=2000,
        train_prevalence=[0.5, 0.5],
        prevalence="grid", n_prevalences=11, repeats=3,
        n_features=10, n_informative=6, n_redundant=2,
        class_sep=1.5, random_state=0,
    )
    Xtr = Xtr.astype(np.float32)
    Xs = [Xb.astype(np.float32) for Xb in Xs]

    # HistNetQ — the FEM outputs a raw latent vector; the network normalises it
    # into the histogram's [0, 1] range internally (no final sigmoid needed).
    torch.manual_seed(0)
    fem_h = nn.Sequential(
        nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 8),
    )
    histnet = HistNetQ(
        feature_extractor=fem_h, n_latent_features=8, n_bins=24,
        ff_layers=(128, 64), n_epochs=80, patience=15,
        bag_size=150, n_bags=150, val_bags=30, random_state=0, device="cpu",
    )

    # GMNet — the FEM need not be bounded; a sigmoid projection per latent space
    # is appended internally.
    torch.manual_seed(0)
    fem_g = nn.Sequential(nn.Linear(10, 32), nn.ReLU())
    gmnet = GMNet(
        feature_extractor=fem_g, n_input_features=10, latent_dim=8,
        n_gaussians=32, n_latent=4, ff_layers=(128, 64), cka_lambda=0.01,
        n_epochs=60, patience=12, bag_size=150, n_bags=120, val_bags=30,
        random_state=0, device="cpu",
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    for (name, q), ax, color in zip(
        {"HistNetQ": histnet, "GMNet": gmnet}.items(),
        axes, ["#2a9d8f", "#e76f51"],
    ):
        q.fit(Xtr, ytr)
        pred = np.vstack([q.predict(Xb) for Xb in Xs])
        DiagonalDisplay.from_predictions(
            prevs, pred, ax=ax, color=color, alpha=0.5, s=20,
        )
        mae = float(np.mean(np.abs(pred - prevs)))
        ax.set_title(f"{name}  (MAE = {mae:.3f})")
    fig.suptitle("Symmetric neural quantifiers on shifted test bags", y=0.99)
    fig.tight_layout()

Both clouds hug the :math:`y = x` diagonal across the whole prevalence range —
the network has learned to track prevalence directly, not just at the training
mix. HistNetQ's per-feature histogram is the lighter, interpretable default;
GMNet's joint Gaussians pay off when latent features interact. Swap in your own
FEM (a CNN for images, a transformer encoder for text) to apply either method to
richer data.

.. seealso::

   - :ref:`neural_quantifiers` — the architecture and every hyper-parameter.
   - :ref:`representations` — the differentiable histogram and Gaussian bag
     representations these methods are built on.
   - :ref:`sphx_method_comparison` — the same diagonal-plot read-out for the
     analytical method families.
