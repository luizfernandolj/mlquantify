import torch
import torch.nn as nn
import torch.nn.functional as F

from mlquantify.representations._base import TorchRepresentation


class DifferentiableHistogramRepresentation(TorchRepresentation):
    r"""Differentiable hard-histogram bag representation (HistNetQ).

    Computes, for every feature dimension, a histogram with learnable bin
    centers and widths. The histogram values are averaged over all examples
    in the bag, making the representation permutation-invariant.

    This is the "hard variant" from Pérez-Mon et al. (2024) (the hard-binning
    differentiable histogram of Yusuf et al., 2020). For an example value
    :math:`v` in feature :math:`k` and bin :math:`b` with center :math:`\mu` and
    half-width :math:`w`, the membership is

    .. math::

        \phi(v; \mu, w) =
        \begin{cases}
            1.01^{\,w - |v - \mu|} & \text{if } 1.01^{\,w - |v - \mu|} > 1 \\
            0 & \text{otherwise,}
        \end{cases}

    i.e. an example contributes to a bin when it lies within the bin half-width,
    :math:`|v - \mu| < w`, and the contribution is the (near-one) value
    :math:`1.01^{\,w-|v-\mu|}` rather than a hard ``1``. This is exactly a
    ``torch.nn.Threshold(1, 0)`` applied to :math:`1.01^{\,w-|v-\mu|}`: the
    threshold zeroes out-of-bin values while the in-bin values keep their
    continuous, **differentiable** form, so gradients flow to the bin centers,
    widths *and* the upstream feature extractor — no straight-through estimator
    is needed. The histogram value is the per-bag mean over examples, a *density*
    that factors out the bag size and is permutation-invariant.

    .. note::

       Inputs are expected to lie in ``[0, 1]`` (the range of the initial bin
       centers). As the BRM of :class:`~mlquantify.neural.HistNetQ`, the network
       squashes the feature-extractor output with a sigmoid before this layer.

    Parameters
    ----------
    n_features : int
        Number of input feature dimensions.
    n_bins : int, default=32
        Number of histogram bins per feature.

    Attributes
    ----------
    output_size : int
        ``n_bins * n_features``
    mu : nn.Parameter of shape (n_bins, n_features)
        Learnable bin centers, initialised to the bin midpoints
        ``(b + 0.5) / n_bins`` in ``[0, 1]``.
    width : nn.Parameter of shape (n_bins, n_features)
        Learnable bin half-widths, initialised to ``1/(2*n_bins) + 0.001``.

    References
    ----------
    Pérez-Mon et al. (2024). Quantification using permutation-invariant
    networks based on histograms. *Neural Computing and Applications*.

    Yusuf, I., Igwegbe, G., & Azeez, O. (2020). Differentiable Histogram with
    Hard-Binning. *arXiv:2012.06311*.
    """

    def __init__(self, n_features: int, n_bins: int = 32):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        # Bin centers at the bin midpoints (b + 0.5) / n_bins in [0, 1].
        centers = (torch.arange(n_bins).float() + 0.5) / n_bins
        self.mu = nn.Parameter(centers.unsqueeze(1).repeat(1, n_features).clone())
        # Half-widths: bin half-spacing plus a small slack (matches the authors).
        init_w = 1.0 / (2 * n_bins) + 0.001
        self.width = nn.Parameter(torch.full((n_bins, n_features), float(init_w)))

    @property
    def output_size(self) -> int:
        return self.n_bins * self.n_features

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute histogram representation for a bag.

        Parameters
        ----------
        X : torch.Tensor of shape (n_examples, n_features) or \
            (n_bags, n_examples, n_features)
            A single bag, or a batch of bags processed together.

        Returns
        -------
        hist : torch.Tensor of shape (n_bins * n_features,) or \
            (n_bags, n_bins * n_features)
            One descriptor per bag; the leading batch axis is present only when
            the input had one.
        """
        # Accept a single bag (n, d) or a batch of bags (B, n, d). The example
        # axis is always the second-to-last; mu/width are (n_bins, d).
        batched = X.dim() == 3
        if not batched:
            X = X.unsqueeze(0)                                    # (1, n, d)
        # X: (B, n, d) -> (B, n, 1, d); mu/width: (n_bins, d) broadcast.
        diff = torch.abs(X.unsqueeze(2) - self.mu)               # (B, n, n_bins, d)
        inside = self.width - diff                               # (B, n, n_bins, d)
        # 1.01^inside is > 1 inside the bin (inside > 0) and < 1 outside; the
        # threshold keeps the differentiable in-bin value and zeroes the rest.
        # ``F.threshold`` is exactly the authors' ``nn.Threshold(1, 0)`` and
        # avoids the extra full-size temporaries a ``torch.where`` would allocate.
        t = torch.pow(1.01, inside)
        phi = F.threshold(t, 1.0, 0.0)                           # Threshold(1, 0)
        # Average over examples (density), then flatten the (n_bins, d) grid.
        out = phi.mean(dim=1).reshape(X.shape[0], -1)            # (B, n_bins * d)
        return out if batched else out[0]
