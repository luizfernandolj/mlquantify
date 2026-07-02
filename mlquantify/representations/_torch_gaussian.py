import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mlquantify.representations._base import TorchRepresentation


class GaussianRepresentation(TorchRepresentation):
    r"""Gaussian latent-space bag representation (GMNet).

    Models the latent space of a bag using K learnable multivariate Gaussian
    distributions with **full covariance**. For every example the likelihood
    under each Gaussian is computed; the K likelihoods are normalised
    (BatchNorm) and averaged over the bag, giving a K-dimensional descriptor per
    latent space. L independent latent spaces are used in parallel (each with
    its own FEM projection and its own K Gaussians); the final representation is
    the concatenation of all L sub-representations.

    This is a faithful port of the authors' ``GMLayer`` (Pérez-Mon et al., 2025):
    the full covariance is what lets the Gaussians capture *feature correlations*
    a per-feature histogram cannot. Positive-definiteness is guaranteed by
    parameterising each covariance through a lower-triangular Cholesky factor
    :math:`L` (so :math:`\Sigma = L L^\top`), which avoids the paper's
    ``geotorch`` dependency while being fully differentiable.

    The Gaussian log-likelihood is the standard multivariate normal,

    .. math::

        \log p(z\mid k) = -\tfrac{1}{2}\left[(z-\mu_k)^\top \Sigma_k^{-1} (z-\mu_k)
        + d\,\log(2\pi) + \log|\Sigma_k|\right].

    Parameters
    ----------
    latent_dim : int
        Dimensionality d of each latent space. The full covariance is
        ``d × d`` per Gaussian, so keep d small (the paper uses ~3–8).
    n_gaussians : int, default=100
        Number K of Gaussian distributions per latent space.
    n_latent : int, default=9
        Number L of independent latent spaces.

    Attributes
    ----------
    output_size : int
        ``n_gaussians * n_latent``
    mu : nn.Parameter of shape (n_latent, n_gaussians, latent_dim)
        Learnable Gaussian centers, initialised uniformly in ``[0, 1]``.
    chol_raw : nn.Parameter of shape (n_latent, n_gaussians, latent_dim, latent_dim)
        Unconstrained parameterisation of the Cholesky factor of each covariance
        (strict lower triangle used directly; diagonal passed through softplus to
        stay positive). Initialised so each covariance is ``c_l · I`` with ``c_l``
        from the paper's Eq. (9) pairwise-distance heuristic,
        :math:`c_l = (\operatorname{mean}_k \min_{j\neq k}\lVert\mu_k-\mu_j\rVert / 2)^2`.
    norm : nn.ModuleList of BatchNorm1d
        One ``BatchNorm1d(K)`` per latent space — the "Norm" of the paper's
        "Norm & Mean" aggregation, applied to the per-Gaussian likelihoods.

    References
    ----------
    Pérez-Mon et al. (2025). Quantification Via Gaussian Latent Space
    Representations. arXiv:2501.13638.
    """

    def __init__(
        self,
        latent_dim: int,
        n_gaussians: int = 100,
        n_latent: int = 9,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_gaussians = n_gaussians
        self.n_latent = n_latent
        L, K, d = n_latent, n_gaussians, latent_dim

        # Gaussian centers, uniform in [0, 1]^d.
        mu = torch.rand(L, K, d)
        self.mu = nn.Parameter(mu)

        # Full covariance via a lower-triangular Cholesky factor: Sigma = L L^T.
        # Initialise Sigma = c_l * I per latent space, with c_l from the paper's
        # Eq. (9) distance heuristic -> Cholesky factor = sqrt(c_l) * I.
        chol = torch.zeros(L, K, d, d)
        eye = torch.eye(d)
        for l in range(L):
            dists = torch.cdist(mu[l], mu[l])              # (K, K)
            dists.fill_diagonal_(float("inf"))
            c = (dists.min(dim=1).values.mean() / 2.0) ** 2
            std = torch.sqrt(c.clamp_min(1e-6))
            # store the raw so that softplus(diag(raw)) == std (inverse softplus).
            diag_raw = torch.log(torch.expm1(std.clamp_min(1e-4)))
            chol[l] = eye * diag_raw
        self.chol_raw = nn.Parameter(chol)

        # "Norm" step: BatchNorm over the K Gaussian likelihoods, per latent space.
        self.norm = nn.ModuleList([nn.BatchNorm1d(K) for _ in range(L)])

    @property
    def output_size(self) -> int:
        return self.n_gaussians * self.n_latent

    def _cholesky(self):
        """Return the lower-triangular Cholesky factors and their diagonals."""
        raw = self.chol_raw                                    # (L, K, d, d)
        lower = torch.tril(raw, diagonal=-1)
        diag = F.softplus(torch.diagonal(raw, dim1=-2, dim2=-1)) + 1e-4  # (L, K, d)
        chol = lower + torch.diag_embed(diag)
        return chol, diag

    def _likelihoods(self, X: torch.Tensor) -> torch.Tensor:
        """Multivariate-normal likelihood of every example under every Gaussian.

        Parameters
        ----------
        X : torch.Tensor of shape (n_bags, n_latent, n_examples, latent_dim)

        Returns
        -------
        lik : torch.Tensor of shape (n_bags, n_latent, n_examples, n_gaussians)
        """
        _, L, _, d = X.shape
        K = self.n_gaussians
        chol, diag = self._cholesky()                          # (L,K,d,d), (L,K,d)
        cov_inv = torch.cholesky_inverse(
            chol.reshape(L * K, d, d)
        ).reshape(L, K, d, d)                                  # Sigma^{-1}
        logdet = 2.0 * torch.log(diag).sum(dim=-1)             # (L, K) = log|Sigma|
        # diff: (B, L, n, K, d)
        diff = X.unsqueeze(3) - self.mu.unsqueeze(0).unsqueeze(2)
        # Mahalanobis term (B, L, n, K) via the authors' einsum contraction.
        maha = torch.einsum("blnki,lkij,blnkj->blnk", diff, cov_inv, diff)
        log_p = -0.5 * (maha + d * float(np.log(2 * np.pi)) + logdet.view(1, L, 1, K))
        return torch.exp(log_p)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian representation for a bag.

        The input X has already been passed through a FEM projection
        (sigmoid-activated) to lie in [0, 1]^latent_dim for each latent space.

        Parameters
        ----------
        X : torch.Tensor of shape (n_latent, n_examples, latent_dim) or \
            (n_bags, n_latent, n_examples, latent_dim)
            Latent projections of the bag examples across all L latent spaces,
            for a single bag or a batch of bags.

        Returns
        -------
        repr : torch.Tensor of shape (n_gaussians * n_latent,) or \
            (n_bags, n_gaussians * n_latent)
            The leading batch axis is present only when the input had one.
        """
        batched = X.dim() == 4
        if not batched:
            X = X.unsqueeze(0)                                 # (1, L, n, d)
        B, L, n, d = X.shape
        K = self.n_gaussians

        lik = self._likelihoods(X)                             # (B, L, n, K)

        # "Norm & Mean": BatchNorm the K per-Gaussian likelihoods (across all
        # examples of the batch), then average over the examples in each bag.
        reps = []
        for l in range(L):
            z = lik[:, l].reshape(B * n, K)                    # (B*n, K)
            z = self.norm[l](z).reshape(B, n, K)
            reps.append(z.mean(dim=1))                         # (B, K)
        out = torch.cat(reps, dim=1)                           # (B, L*K)
        return out if batched else out[0]
