import operator

import numpy as np
from sklearn.base import BaseEstimator

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - runtime-dependent
    torch = None


class TorchClassifierWrapper(BaseEstimator):
    """Wrap a torch nn.Module with a sklearn-compatible classifier interface.

    Provides ``fit``, ``predict_proba``, and ``transform`` so that any
    custom PyTorch model can be used as the ``estimator`` in
    :class:`~mlquantify.neural.QuaNet`.

    Parameters
    ----------
    module : nn.Module
        Full PyTorch model (encoder → classification head).
    encoder_attr : str or None, default=None
        Dotted attribute path to the sub-module used for embeddings,
        e.g. ``"encoder"`` or ``"backbone.encoder"``.
        If ``None``, ``transform`` calls ``forward`` on the full module
        (the module must then expose a ``encode`` method or equivalent).
        If a string, the sub-module is retrieved via ``getattr`` (supports
        dotted paths) and its ``forward`` is called for ``transform``.
    n_classes : int or None, default=None
        Number of output classes. Inferred from data during ``fit`` if None.
    lr : float, default=1e-3
        Learning rate for Adam optimiser used in ``fit``.
    n_epochs : int, default=20
        Training epochs for ``fit``.
    batch_size : int, default=64
        Mini-batch size for ``fit``.
    device : str, default='cpu'
        PyTorch device string.

    Examples
    --------
    Standard usage: the module has a named encoder sub-module.

    .. code-block:: python

        import torch.nn as nn
        from mlquantify.neural import TorchClassifierWrapper

        class MyCatDogNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(512, 128), nn.ReLU())
                self.head    = nn.Linear(128, 2)
            def forward(self, x):
                return self.head(self.encoder(x))

        wrapper = TorchClassifierWrapper(MyCatDogNet(), encoder_attr='encoder')
        wrapper.fit(X_train, y_train)
        wrapper.predict_proba(X_test)   # shape (n, 2)
        wrapper.transform(X_test)       # shape (n, 128)  — encoder output
    """

    def __init__(
        self,
        module,
        encoder_attr=None,
        n_classes=None,
        lr=1e-3,
        n_epochs=20,
        batch_size=64,
        device="cpu",
    ):
        if torch is None:
            raise ImportError("PyTorch is required to use TorchClassifierWrapper.")
        self.module = module
        self.encoder_attr = encoder_attr
        self.n_classes = n_classes
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

    def fit(self, X, y):
        """Train the module with cross-entropy loss.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        import torch.utils.data as data_utils

        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.long)
        self.classes_ = np.unique(y)

        dataset = data_utils.TensorDataset(X_t, y_t)
        loader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.module.to(self.device)
        self.module.train()
        optimiser = torch.optim.Adam(self.module.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(self.n_epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                out = self.module(xb)
                criterion(out, yb).backward()
                optimiser.step()

        self.module.eval()
        return self

    def predict_proba(self, X):
        """Return class posterior probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
        """
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        self.module.eval()
        with torch.no_grad():
            logits = self.module(X_t)
            proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    def transform(self, X):
        """Return encoder embeddings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        embeddings : ndarray of shape (n_samples, embed_dim)
        """
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(self.device)
        if self.encoder_attr is not None:
            encoder = operator.attrgetter(self.encoder_attr)(self.module)
        else:
            encoder = self.module
        encoder.eval()
        with torch.no_grad():
            emb = encoder(X_t).cpu().numpy()
        return emb
