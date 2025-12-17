import os
import random
from typing import Dict, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import (
    AggregationMixin,
    SoftLearnerQMixin,
    get_aggregation_requirements
)
from mlquantify.utils import (
    validate_y,
    validate_data,
)
from mlquantify.utils._validation import validate_prevalences
from mlquantify.model_selection import APP
from mlquantify.utils import get_prev_from_labels
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils import _fit_context

from mlquantify.adjust_counting import CC, AC, PCC, PAC

EPS = 1e-12


class QuaNetModule(nn.Module):
    r"""
    PyTorch module implementing the forward pass of QuaNet, as described in
    Esuli et al. (2018) "A Recurrent Neural Network for Sentiment Quantification". [file:1][file:3]

    This module takes as input:
      - the document embeddings of a bag,
      - the posterior probabilities for each document in the bag,
      - a fixed-size vector of quantification statistics (e.g., CC/ACC/PCC/PACC outputs),

    and outputs a class-prevalence vector for the bag.

    Core idea:
      - Concatenate document embeddings and posterior probabilities.
      - Sort the sequence by the posterior probability of a selected class (optional).
      - Pass the sequence through an LSTM (possibly bidirectional).
      - Take the final hidden state(s) as a "quantification embedding".
      - Concatenate this embedding with the quantification statistics.
      - Pass through one or more fully connected layers and a final softmax to obtain prevalences.
    """

    def __init__(
        self,
        doc_embedding_size: int,
        n_classes: int,
        stats_size: int,
        lstm_hidden_size: int = 64,
        lstm_nlayers: int = 1,
        ff_layers: Sequence[int] = (1024, 512),
        bidirectional: bool = True,
        qdrop_p: float = 0.5,
        order_by: int | None = 0,
    ) -> None:
        """
        Parameters
        ----------
        doc_embedding_size : int
            Dimensionality of document embeddings (output of `learner.transform`).
        n_classes : int
            Number of classes of the quantification problem.
        stats_size : int
            Dimensionality of the statistics vector concatenated to the LSTM embedding
            (e.g. concatenated prevalence estimates from CC, ACC, PCC, PACC, EMQ, ...).
        lstm_hidden_size : int, default=64
            Hidden size of the LSTM cell(s).
        lstm_nlayers : int, default=1
            Number of stacked LSTM layers.
        ff_layers : sequence of int, default=(1024, 512)
            Sizes of the fully connected layers on top of the quantification embedding.
        bidirectional : bool, default=True
            Whether to use a bidirectional LSTM.
        qdrop_p : float, default=0.5
            Dropout probability used in the LSTM and in the fully connected layers.
        order_by : int or None, default=0
            Index of the class whose posterior probability is used for sorting the sequence.
            If None, no sorting is performed.
        """
        super().__init__()

        self.n_classes = n_classes
        self.order_by = order_by
        self.hidden_size = lstm_hidden_size
        self.nlayers = lstm_nlayers
        self.bidirectional = bidirectional
        self.ndirections = 2 if bidirectional else 1
        self.qdrop_p = qdrop_p

        # LSTM input: [embedding, posterior_probs]
        self.lstm = nn.LSTM(
            input_size=doc_embedding_size + n_classes,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_nlayers,
            bidirectional=bidirectional,
            dropout=qdrop_p if lstm_nlayers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(self.qdrop_p)

        lstm_output_size = self.hidden_size * self.ndirections
        ff_input_size = lstm_output_size + stats_size

        prev_size = ff_input_size
        self.ff_layers = nn.ModuleList()
        for lin_size in ff_layers:
            self.ff_layers.append(nn.Linear(prev_size, lin_size))
            prev_size = lin_size

        self.output = nn.Linear(prev_size, n_classes)

    @property
    def device(self) -> torch.device:
        """Return the device on which the module parameters are stored."""
        return next(self.parameters()).device

    def _init_hidden(self, batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden and cell states with zeros.

        Parameters
        ----------
        batch_size : int
            Batch size for which the hidden state is initialized.

        Returns
        -------
        (h0, c0) : (Tensor, Tensor)
            Initial hidden and cell states.
        """
        directions = 2 if self.bidirectional else 1
        h = torch.zeros(self.nlayers * directions, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.nlayers * directions, batch_size, self.hidden_size, device=self.device)
        return h, c

    def forward(
        self,
        doc_embeddings: np.ndarray | torch.Tensor,
        doc_posteriors: np.ndarray | torch.Tensor,
        statistics: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of QuaNet.

        Parameters
        ----------
        doc_embeddings : array-like of shape (n_docs, emb_dim)
            Document embeddings of all items in the bag.
        doc_posteriors : array-like of shape (n_docs, n_classes)
            Posterior probabilities `P(y | x)` for each document in the bag, produced by the base classifier.
        statistics : array-like of shape (stats_size,) or (1, stats_size)
            Vector of quantification-related statistics (e.g., CC/ACC/PCC/PACC estimates, TPR/FPR, etc.).

        Returns
        -------
        prevalence : torch.Tensor of shape (1, n_classes)
            Estimated class-prevalence vector for the input bag.
        """
        device = self.device

        if not isinstance(doc_embeddings, torch.Tensor):
            doc_embeddings = torch.as_tensor(doc_embeddings, dtype=torch.float32, device=device)
        else:
            doc_embeddings = doc_embeddings.to(device)

        if not isinstance(doc_posteriors, torch.Tensor):
            doc_posteriors = torch.as_tensor(doc_posteriors, dtype=torch.float32, device=device)
        else:
            doc_posteriors = doc_posteriors.to(device)

        if not isinstance(statistics, torch.Tensor):
            statistics = torch.as_tensor(statistics, dtype=torch.float32, device=device)
        else:
            statistics = statistics.to(device)

        # Optional sorting by posterior of a specific class
        if self.order_by is not None:
            order = torch.argsort(doc_posteriors[:, self.order_by])
            doc_embeddings = doc_embeddings[order]
            doc_posteriors = doc_posteriors[order]

        # Sequence of concatenated embeddings and posteriors
        embedded_posteriors = torch.cat((doc_embeddings, doc_posteriors), dim=-1)  # (n_docs, emb_dim + n_classes)
        embedded_posteriors = embedded_posteriors.unsqueeze(0)  # (1, n_docs, emb_dim + n_classes)

        self.lstm.flatten_parameters()
        _, (rnn_hidden, _) = self.lstm(embedded_posteriors, self._init_hidden(batch_size=1))
        # rnn_hidden: (num_layers * num_directions, batch=1, hidden_size)
        rnn_hidden = rnn_hidden.view(self.nlayers, self.ndirections, 1, self.hidden_size)
        # Take the first layer's hidden states, flatten directions
        quant_embedding = rnn_hidden[0].view(-1)  # (hidden_size * ndirections,)

        if statistics.dim() == 1:
            statistics = statistics.view(-1)

        # Concatenate LSTM quantification embedding with statistics
        quant_embedding = torch.cat((quant_embedding, statistics), dim=0)

        x = quant_embedding.unsqueeze(0)
        for linear in self.ff_layers:
            x = self.dropout(F.relu(linear(x)))

        logits = self.output(x).view(1, -1)
        prevalence = torch.softmax(logits, dim=-1)
        return prevalence


class QuaNet(SoftLearnerQMixin, AggregationMixin, BaseQuantifier):
    r"""
    QuaNetQuantifier: a deep quantification method following the QuaNet architecture,
    implemented in the `mlquantify` style.

    This class wraps a base probabilistic learner that:
      - can be trained on labeled instances,
      - can output posterior probabilities via `predict_proba(X)`,
      - can transform instances into embeddings via `transform(X)`.

    QuaNet then learns a mapping from bags of instances to class-prevalence vectors by:
      - generating artificial bags using the APP protocol (APP: Artificial Prevalence Protocol),
      - computing simple quantification estimates (CC, ACC, PCC, PACC, ...) on each bag,
      - feeding the sequence of (embedding, posterior) pairs and the statistics vector into an LSTM-based network,
      - minimizing a bag-level quantification loss (MSE between predicted and true prevalences).[file:1][file:3]

    Parameters
    ----------
    learner : estimator
        Base probabilistic classifier. Must implement:
          - fit(X, y),
          - predict_proba(X) -> array-like (n_samples, n_classes),
          - transform(X) -> array-like (n_samples, emb_dim).
    fit_learner : bool, default=True
        If True, the learner is trained inside QuaNetQuantifier.fit.
        If False, it is assumed to be already fitted.
    sample_size : int, default=100
        Bag size used by the APP protocol during QuaNet training.
    n_epochs : int, default=100
        Maximum number of QuaNet training epochs.
    tr_iter_per_epoch : int, default=500
        Number of APP samplings (training iterations) per epoch.
    va_iter_per_epoch : int, default=100
        Number of APP samplings (validation iterations) per epoch.
    lr : float, default=1e-3
        Learning rate for the Adam optimizer.
    lstm_hidden_size : int, default=64
        Hidden size of the QuaNet LSTM.
    lstm_nlayers : int, default=1
        Number of layers in the QuaNet LSTM.
    ff_layers : sequence of int, default=(1024, 512)
        Sizes of the fully connected layers on top of the LSTM quantification embedding.
    bidirectional : bool, default=True
        Whether to use a bidirectional LSTM.
    qdrop_p : float, default=0.5
        Dropout probability used in QuaNet.
    patience : int, default=10
        Early-stopping patience in number of epochs without validation improvement.
    checkpointdir : str, default="./checkpoint_quanet"
        Directory where intermediate QuaNet weights are stored.
    checkpointname : str or None, default=None
        Name of the saved checkpoint file. If None, a random name is generated.
    device : {"cpu", "cuda"}, default="cuda"
        Device on which to run the QuaNet model.
    """

    _parameter_constraints = {
        "fit_learner": [Interval(0, None, inclusive_left=False), Options([None])],
        "sample_size": [Interval(0, None, inclusive_left=False), Options([None])],
        "n_epochs": [Interval(0, None, inclusive_left=False), Options([None])],
        "tr_iter_per_epoch": [Interval(0, None, inclusive_left=False), Options([None])],
        "va_iter_per_epoch": [Interval(0, None, inclusive_left=False), Options([None])],
        "lr": [Interval(0, None, inclusive_left=False), Options([None])],
        "lstm_hidden_size": [Interval(0, None, inclusive_left=False), Options([None])],
        "lstm_nlayers": [Interval(0, None, inclusive_left=False), Options([None])],
        "ff_layers": [Interval(0, None, inclusive_left=False), Options([None])],
        "bidirectional": [Interval(0, None, inclusive_left=False), Options([None])],
        "qdrop_p": [Interval(0, None, inclusive_left=False), Options([None])],
        "patience": [Interval(0, None, inclusive_left=False), Options([None])],
        "checkpointdir": "string",
        "checkpointname": "string",
        "device": "string",
    }


    def __init__(
        self,
        learner,
        fit_learner: bool = True,
        sample_size: int = 100,
        n_epochs: int = 100,
        tr_iter_per_epoch: int = 500,
        va_iter_per_epoch: int = 100,
        lr: float = 1e-3,
        lstm_hidden_size: int = 64,
        lstm_nlayers: int = 1,
        ff_layers: Sequence[int] = (1024, 512),
        bidirectional: bool = True,
        qdrop_p: float = 0.5,
        patience: int = 10,
        checkpointdir: str = "./checkpoint_quanet",
        checkpointname: str | None = None,
        device: str = "cuda",
    ) -> None:

        assert hasattr(learner, "transform"), ...
        assert hasattr(learner, "predict_proba"), ...

        # save hyperparameters as attributes
        self.learner = learner
        self.fit_learner = fit_learner
        self.sample_size = sample_size
        self.n_epochs = n_epochs
        self.tr_iter_per_epoch = tr_iter_per_epoch
        self.va_iter_per_epoch = va_iter_per_epoch
        self.lr = lr
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_nlayers = lstm_nlayers
        self.ff_layers = ff_layers
        self.bidirectional = bidirectional      # <-- IMPORTANT
        self.qdrop_p = qdrop_p
        self.patience = patience
        self.checkpointdir = checkpointdir
        self.checkpointname = checkpointname
        self.device = torch.device(device)

        self.quanet_params: Dict[str, Any] = dict(
            lstm_hidden_size=lstm_hidden_size,
            lstm_nlayers=lstm_nlayers,
            ff_layers=ff_layers,
            bidirectional=bidirectional,
            qdrop_p=qdrop_p,
        )

        os.makedirs(self.checkpointdir, exist_ok=True)
        if self.checkpointname is None:
            local_random = random.Random()
            random_code = "-".join(str(local_random.randint(0, 1_000_000)) for _ in range(5))
            self.checkpointname = f"QuaNet-{random_code}"
        self.checkpoint = os.path.join(self.checkpointdir, self.checkpointname)

        self._classes_ = None
        self.quantifiers = {}
        self.quanet = None
        self.optim = None

        self.status: Dict[str, float] = {
            "tr-loss": -1.0,
            "va-loss": -1.0,
            "tr-mae": -1.0,
            "va-mae": -1.0,
        }

    @property
    def classes_(self) -> np.ndarray:
        """Return the class labels observed during training."""
        return self._classes_

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, learner_fitted: bool = False):
        """
        Train QuaNet on labeled instances.

        The procedure is:
          1. Optionally split data into:
             - a portion for training the base learner (if `fit_learner=True`),
             - a portion for QuaNet training,
             - a portion for QuaNet validation.
          2. Train the base learner if requested.
          3. Compute posterior probabilities and embeddings for the QuaNet train/valid sets.
          4. Train simple aggregative quantifiers (CC, ACC, PCC, PACC) on the validation set.
          5. Initialize QuaNetModule and train it using APP-generated bags, minimizing MSE
             between predicted and true prevalences.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances.
        y : array-like of shape (n_samples,)
            Class labels.
        learner_fitted : bool, default=False
            If True, the learner is assumed to already be fitted and will not be retrained.

        Returns
        -------
        self : QuaNetQuantifier
            The fitted quantifier.
        """
        X, y = validate_data(self, X, y, ensure_2d=True, ensure_min_samples=2)
        self._classes_ = np.unique(y)

        n_samples = X.shape[0]
        rng = np.random.RandomState(42)
        idx_all = np.arange(n_samples)
        rng.shuffle(idx_all)

        # Split scheme similar in spirit to QuaPy:
        # - If fit_learner: 40% learner training, 40% QuaNet training, 20% QuaNet validation.
        # - Else: 66% QuaNet training, 34% QuaNet validation.
        if self.fit_learner and not learner_fitted:
            n_clf = int(0.4 * n_samples)
            clf_idx = idx_all[:n_clf]
            rest_idx = idx_all[n_clf:]
            n_train = int(0.66 * rest_idx.shape[0])
            train_idx = rest_idx[:n_train]
            valid_idx = rest_idx[n_train:]

            X_clf, y_clf = X[clf_idx], y[clf_idx]
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[valid_idx], y[valid_idx]

            # Train base learner on classifier data
            self.learner.fit(X_clf, y_clf)
        else:
            n_train = int(0.66 * n_samples)
            train_idx = idx_all[:n_train]
            valid_idx = idx_all[n_train:]
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, y_valid = X[valid_idx], y[valid_idx]

        # Posterior probabilities and embeddings for QuaNet train/valid sets
        train_post = self.learner.predict_proba(X_train)
        valid_post = self.learner.predict_proba(X_valid)
        train_embed = self.learner.transform(X_train)
        valid_embed = self.learner.transform(X_valid)

        n_classes = len(self._classes_)

        # Train simple aggregative quantifiers on the validation set (to produce statistics)[file:1][file:3]
        self.quantifiers = {
            "cc": CC(self.learner, learner_fitted=True),
            "acc": AC(self.learner, learner_fitted=True),
            "pcc": PCC(self.learner, learner_fitted=True),
            "pacc": PAC(self.learner, learner_fitted=True),
        }
        for q in self.quantifiers.values():
            q.fit(X_valid, y_valid, learner_fitted=True)

        nQ = len(self.quantifiers)
        stats_size = nQ * n_classes

        # In the binary case it is common to sort by the positive class score (index 0 or 1).
        # Here we choose index 0; for multi-class we disable sorting by default.
        order_by = 0 if n_classes == 2 else None

        # Initialize QuaNet module and optimizer
        self.quanet = QuaNetModule(
            doc_embedding_size=train_embed.shape[1],
            n_classes=n_classes,
            stats_size=stats_size,
            order_by=order_by,
            **self.quanet_params,
        ).to(self.device)
        self.optim = torch.optim.Adam(self.quanet.parameters(), lr=self.lr)

        best_va_loss = np.inf
        best_epoch = -1
        patience_left = self.patience

        # Training loop with early stopping
        for epoch_i in range(1, self.n_epochs + 1):
            self._epoch(
                X_train, y_train, train_embed, train_post,
                iterations=self.tr_iter, epoch=epoch_i, train=True
            )
            self._epoch(
                X_valid, y_valid, valid_embed, valid_post,
                iterations=self.va_iter, epoch=epoch_i, train=False
            )

            va_loss = self.status["va-loss"]
            if va_loss < best_va_loss - 1e-6:
                best_va_loss = va_loss
                best_epoch = epoch_i
                patience_left = self.patience
                torch.save(self.quanet.state_dict(), self.checkpoint)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    # Early stopping: restore best model
                    self.quanet.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
                    break

        return self

    def _get_aggregative_estims(
        self,
        posteriors: np.ndarray,
        train_predictions: np.ndarray | None = None,
        train_y_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute a vector of simple quantification estimates (statistics) for a given bag.

        This method inspects each quantifier in `self.quantifiers` (e.g., CC, ACC, PCC, PACC)
        and automatically adapts the call to its `aggregate` method according to its declared
        aggregation requirements. Requirements are obtained via
        `mlquantify.base_aggregative.get_aggregation_requirements`, which specifies whether
        a quantifier needs:

          - only test predictions,
          - test predictions + training labels,
          - or test predictions + training posterior probabilities + training labels.

        The resulting prevalence estimates from all quantifiers are concatenated into a single
        statistics vector.

        Parameters
        ----------
        posteriors : ndarray of shape (n_docs, n_classes)
            Posterior probabilities for the documents in the bag (test predictions).
        train_predictions : ndarray of shape (n_train, n_classes), optional
            Posterior probabilities for the training instances (used by some aggregative methods).
            If None and required by some quantifier, a ValueError is raised.
        train_y_values : ndarray of shape (n_train,), optional
            Training labels (used by some aggregative methods). If None and required by some
            quantifier, a ValueError is raised.

        Returns
        -------
        stats : ndarray of shape (n_quantifiers * n_classes,)
            Concatenated prevalence estimates from all simple quantifiers.
        """
        label_predictions = np.argmax(posteriors, axis=-1)
        stats_list: list[float] = []

        for q in self.quantifiers.values():
            reqs = get_aggregation_requirements(q)

            # Determine which arguments to pass based on requirements
            if reqs.requires_train_proba and reqs.requires_train_labels:
                if train_predictions is None or train_y_values is None:
                    raise ValueError(
                        f"Quantifier {q.__class__.__name__} requires training probabilities "
                        "and training labels, but they were not provided."
                    )
                prev = q.aggregate(posteriors, train_predictions, train_y_values)
            elif reqs.requires_train_labels:
                if train_y_values is None:
                    raise ValueError(
                        f"Quantifier {q.__class__.__name__} requires training labels, "
                        "but train_y_values was not provided."
                    )
                prev = q.aggregate(posteriors, train_y_values)
            else:
                # Only test predictions are required; some quantifiers expect hard labels,
                # others soft probabilities. We rely on their `aggregate` signature to decide.
                try:
                    prev = q.aggregate(posteriors)
                except TypeError:
                    # If the quantifier expects hard labels instead of probabilities, try again
                    prev = q.aggregate(label_predictions)

            # `prev` is usually a 1D array of length n_classes
            prev = np.asarray(prev, dtype=np.float32)
            stats_list.extend(prev)

        return np.asarray(stats_list, dtype=np.float32)

    def _epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        embeddings: np.ndarray,
        posteriors: np.ndarray,
        iterations: int,
        epoch: int,
        train: bool,
    ) -> None:
        """
        Run one training or validation epoch of QuaNet using APP-generated bags.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Instances that form the universe for bag sampling in this epoch.
        y : array-like of shape (n_samples,)
            Labels corresponding to X.
        embeddings : array-like of shape (n_samples, emb_dim)
            Precomputed document embeddings for X.
        posteriors : array-like of shape (n_samples, n_classes)
            Precomputed posterior probabilities for X.
        iterations : int
            Number of bags (APP samples) to draw in this epoch.
        epoch : int
            Epoch index, used only for logging.
        train : bool
            True for training epoch, False for validation epoch.
        """
        assert self.quanet is not None
        mse_loss = nn.MSELoss()
        self.quanet.train(mode=train)

        losses = []
        mae_errors = []

        # APP protocol: generates indices of bags with controlled prevalences.[file:2][file:3]
        app = APP(
            batch_size=self.sample_size,
            n_prevalences=5,
            repeats=iterations,
            random_state=None if train else 0,
        )

        # We use APP.split to iterate over indices of each bag
        idx_iter = app.split(X, y)
        pbar = tqdm(range(iterations), desc=f"[QuaNet] epoch {epoch} ({'train' if train else 'val'})")

        for _ in pbar:
            try:
                idx = next(idx_iter)
            except StopIteration:
                break

            idx = np.asarray(idx)
            X_bag = X[idx]
            y_bag = y[idx]
            embed_bag = embeddings[idx]
            post_bag = posteriors[idx]

            quant_estims = self._get_aggregative_estims(
                post_bag,
                train_predictions=post_bag,   # train_proba = posteriors of this bag
                train_y_values=y_bag,         # train_labels = labels of this bag
            )
            ptrue_np = get_prev_from_labels(y_bag, classes=self._classes_)  # true prevalences
            ptrue = torch.as_tensor(ptrue_np[None, :], dtype=torch.float32, device=self.device)

            if train:
                assert self.optim is not None
                self.optim.zero_grad()
                phat = self.quanet(embed_bag, post_bag, quant_estims)
                loss = mse_loss(phat, ptrue)
                mae = torch.mean(torch.abs(phat - ptrue))
                loss.backward()
                self.optim.step()
            else:
                with torch.no_grad():
                    phat = self.quanet(embed_bag, post_bag, quant_estims)
                    loss = mse_loss(phat, ptrue)
                    mae = torch.mean(torch.abs(phat - ptrue))

            losses.append(loss.item())
            mae_errors.append(mae.item())

            mse_val = float(np.mean(losses))
            mae_val = float(np.mean(mae_errors))
            if train:
                self.status["tr-loss"] = mse_val
                self.status["tr-mae"] = mae_val
            else:
                self.status["va-loss"] = mse_val
                self.status["va-mae"] = mae_val

            pbar.set_postfix(
                tr_mse=self.status["tr-loss"],
                va_mse=self.status["va-loss"],
                tr_mae=self.status["tr-mae"],
                va_mae=self.status["va-mae"],
            )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Estimate the class-prevalence vector for a bag of instances X.

        In the typical quantification scenario, X is a set (or bag) of unlabelled items
        drawn from some target distribution, and `predict` returns the estimated prevalence
        of each class within that bag.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Unlabelled instances forming the bag to be quantified.

        Returns
        -------
        prevalence : ndarray of shape (n_classes,)
            Estimated class-prevalence vector for the bag X.
        """
        assert self.quanet is not None, "QuaNet must be fitted before calling predict."
        posteriors = self.learner.predict_proba(X)
        embeddings = self.learner.transform(X)
        quant_estims = self._get_aggregative_estims(posteriors)

        self.quanet.eval()
        with torch.no_grad():
            prevalence = self.quanet(embeddings, posteriors, quant_estims)
            if self.device.type == "cuda":
                prevalence = prevalence.cpu()
            prevalence = prevalence.numpy().flatten()

        prevalence = np.clip(prevalence, EPS, None)
        prevalence = prevalence / prevalence.sum()
        prevalence = validate_prevalences(self, prevalence, self._classes_)
        return prevalence
