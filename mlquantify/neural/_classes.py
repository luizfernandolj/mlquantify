from __future__ import annotations

import os
import random
from typing import Dict, Any, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    from torch.nn import MSELoss
    from torch.nn.functional import relu
except ImportError:
    torch = None

    class _TorchStub:
        class Module:
            pass

    nn = _TorchStub()

    def relu(*_args, **_kwargs):
        raise ImportError("PyTorch is required to use mlquantify.neural.")

    class MSELoss:
        def __init__(self, *_args, **_kwargs):
            raise ImportError("PyTorch is required to use mlquantify.neural.")

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import (
    AggregativeMixin,
    SoftPredictionMixin,
    get_aggregation_requirements,
    _get_estimator_function
)
from mlquantify.utils import (
    validate_y,
    validate_data,
    check_classes_attribute,
)
from mlquantify.utils._validation import validate_prevalences
from mlquantify.model_selection import UPP
from mlquantify.utils import get_prev_from_labels
from mlquantify.utils._constraints import Interval, Options
from mlquantify.utils import _fit_context

from mlquantify.counting import CC, GACC, PCC, GPACC
from mlquantify.likelihood import EMQ
from mlquantify.neural._utils import (
    rae_loss, bag_mixer, cka_regularization, BackgroundPrefetch,
)

EPS = 1e-12



class EarlyStop:
    r"""Early stopping condition for neural network training.

    Tracks a monitored metric across epochs and sets ``STOP = True`` when
    the metric fails to improve for ``patience`` consecutive calls.

    Parameters
    ----------
    patience : int
        Number of consecutive non-improving epochs before stopping.
    lower_is_better : bool, default=True
        If ``True``, a lower metric value is considered better.

    Attributes
    ----------
    best_score : float or None
        Best metric value seen so far.
    best_epoch : int or None
        Epoch at which the best score was recorded.
    STOP : bool
        ``True`` when the stopping condition has been triggered.
    IMPROVED : bool
        ``True`` if the last call improved the best score.

    Examples
    --------
    >>> from mlquantify.neural._classes import EarlyStop
    >>> es = EarlyStop(patience=2, lower_is_better=True)
    >>> es(0.9, epoch=0)
    >>> es(0.7, epoch=1)
    >>> es.IMPROVED
    True
    >>> es(1.0, epoch=2)
    >>> es.STOP
    False
    >>> es(1.0, epoch=3)
    >>> es.STOP
    True
    >>> es.best_epoch
    1
    """

    def __init__(self, patience, lower_is_better=True):

        self.PATIENCE_LIMIT = patience
        self.better = lambda a,b: a<b if lower_is_better else a>b
        self.patience = patience
        self.best_score = None
        self.best_epoch = None
        self.STOP = False
        self.IMPROVED = False

    def __call__(self, watch_score, epoch):
        """Commit the score for a given epoch and update the stopping state.

        If the score improves over the best score seen so far, the patience
        counter is reset; otherwise it is decremented. When the counter reaches
        zero, ``STOP`` is set to ``True``.

        Parameters
        ----------
        watch_score : float
            Metric value to evaluate for the current epoch.
        epoch : int
            Index of the current training epoch.
        """
        self.IMPROVED = (self.best_score is None or self.better(watch_score, self.best_score))
        if self.IMPROVED:
            self.best_score = watch_score
            self.best_epoch = epoch
            self.patience = self.PATIENCE_LIMIT
        else:
            self.patience -= 1
            if self.patience <= 0:
                self.STOP = True



class QuaNetModule(nn.Module):
    r"""PyTorch module implementing the QuaNet forward pass.

    Takes as input a bag of document embeddings, their posterior probabilities,
    and a vector of simple quantification statistics (e.g. CC, PCC, EMQ outputs).
    Passes the (embedding, posterior) sequence through a bidirectional LSTM,
    concatenates the final hidden state with the statistics vector, and produces
    a class-prevalence estimate through fully connected layers with softmax output.

    Parameters
    ----------
    doc_embedding_size : int
        Dimensionality of document embeddings.
    n_classes : int
        Number of target classes.
    stats_size : int
        Size of the statistics vector concatenated after the LSTM.
    lstm_hidden_size : int, default=64
        Hidden size of the LSTM.
    lstm_nlayers : int, default=1
        Number of stacked LSTM layers.
    ff_layers : sequence of int, default=(1024, 512)
        Sizes of fully connected layers after the LSTM embedding.
    bidirectional : bool, default=True
        Whether to use a bidirectional LSTM.
    qdrop_p : float, default=0.5
        Dropout probability in the LSTM and FC layers.
    order_by : int or None, default=0
        Class index used to sort the input sequence by posterior probability.
        ``None`` disables sorting.

    References
    ----------
    .. dropdown:: References

        .. [1] Esuli, A., Moreo, A., & Sebastiani, F. (2018).
               A Recurrent Neural Network for Sentiment Quantification.
               *CIKM*, pp. 1775–1778.
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
            Dimensionality of document embeddings (output of `estimator.transform`).
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
        self.ndirections = 2 if self.bidirectional else 1
        self.qdrop_p = qdrop_p
        self.lstm = torch.nn.LSTM(doc_embedding_size + n_classes,  # +n_classes stands for the posterior probs. (concatenated)
                                  lstm_hidden_size, lstm_nlayers, bidirectional=bidirectional,
                                  dropout=qdrop_p, batch_first=True)
        self.dropout = torch.nn.Dropout(self.qdrop_p)

        lstm_output_size = self.hidden_size * self.ndirections
        ff_input_size = lstm_output_size + stats_size
        prev_size = ff_input_size
        self.ff_layers = torch.nn.ModuleList()
        for lin_size in ff_layers:
            self.ff_layers.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.output = torch.nn.Linear(prev_size, n_classes)

    @property
    def device(self) -> torch.device:
        """Return the device on which the module parameters are stored."""
        return next(self.parameters()).device

    def _init_hidden(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize LSTM hidden and cell states with zeros.

        Returns
        -------
        (h0, c0) : (Tensor, Tensor)
            Initial hidden and cell states.
        """
        directions = 2 if self.bidirectional else 1
        var_hidden = torch.zeros(self.nlayers * directions, 1, self.hidden_size)
        var_cell = torch.zeros(self.nlayers * directions, 1, self.hidden_size)
        if next(self.lstm.parameters()).is_cuda:
            var_hidden, var_cell = var_hidden.cuda(), var_cell.cuda()
        return var_hidden, var_cell

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
        doc_embeddings = torch.as_tensor(doc_embeddings, dtype=torch.float, device=device)
        doc_posteriors = torch.as_tensor(doc_posteriors, dtype=torch.float, device=device)
        statistics = torch.as_tensor(statistics, dtype=torch.float, device=device)

        if self.order_by is not None:
            order = torch.argsort(doc_posteriors[:, self.order_by])
            doc_embeddings = doc_embeddings[order]
            doc_posteriors = doc_posteriors[order]

        embeded_posteriors = torch.cat((doc_embeddings, doc_posteriors), dim=-1)

        # The entire set represents a single quantification instance, so batch_size=1.
        # the shape should be (1, number-of-instances, embedding-size + n_classes)

        embeded_posteriors = embeded_posteriors.unsqueeze(0)

        self.lstm.flatten_parameters()
        _, (rnn_hidden,_) = self.lstm(embeded_posteriors, self._init_hidden())
        rnn_hidden = rnn_hidden.view(self.nlayers, self.ndirections, 1, self.hidden_size)
        quant_embedding = rnn_hidden[0].view(-1)
        quant_embedding = torch.cat((quant_embedding, statistics))

        abstracted = quant_embedding.unsqueeze(0)
        
        for linear in self.ff_layers:
            abstracted = self.dropout(relu(linear(abstracted)))

        logits = self.output(abstracted).view(1, -1)
        prevalence = torch.softmax(logits, -1)

        return prevalence


class QuaNet(SoftPredictionMixin, AggregativeMixin, BaseQuantifier):
    r"""QuaNet: deep neural quantification with an LSTM architecture.

    Learns a mapping from bags of instances to class-prevalence vectors using
    an LSTM network. During training, artificial bags are generated via the APP
    protocol; for each bag the network receives document embeddings, posterior
    probabilities, and simple quantification statistics (CC, PCC, EMQ …) and
    is trained to minimise the MSE against the true bag prevalences.

    Requires a base estimator that implements ``fit``, ``predict_proba``, and
    ``transform`` (the last to produce document embeddings). PyTorch must be
    installed.

    Parameters
    ----------
    estimator : estimator
        Base probabilistic classifier with ``fit``, ``predict_proba``, and
        ``transform`` methods.
    fit_estimator : bool, default=True
        If ``True``, fit the estimator inside :meth:`fit`.
    sample_size : int, default=100
        Bag size used by the APP protocol during training.
    n_epochs : int, default=100
        Maximum number of training epochs.
    tr_iter : int, default=500
        Training APP samplings per epoch.
    va_iter : int, default=100
        Validation APP samplings per epoch.
    lr : float, default=1e-3
        Learning rate for the Adam optimiser.
    lstm_hidden_size : int, default=64
        Hidden size of the LSTM.
    lstm_nlayers : int, default=1
        Number of LSTM layers.
    ff_layers : sequence of int, default=(1024, 512)
        Sizes of the fully connected layers above the LSTM embedding.
    bidirectional : bool, default=True
        Whether to use a bidirectional LSTM.
    qdrop_p : float, default=0.5
        Dropout probability in the network.
    patience : int, default=10
        Early-stopping patience (epochs without validation improvement).
    checkpointdir : str, default='./checkpoint_quanet'
        Directory for saving intermediate model weights.
    checkpointname : str or None, default=None
        Checkpoint filename. ``None`` generates a random name.
    device : {'cpu', 'cuda'}, default='cuda'
        Device used for PyTorch computations.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Class labels seen during ``fit``.

    Examples
    --------
    .. code-block:: python

        # Requires PyTorch and an estimator with a transform() method
        from mlquantify.neural import QuaNet
        q = QuaNet(estimator=my_embedding_classifier, device='cpu')
        q.fit(X_train, y_train)
        q.predict(X_test)

    References
    ----------
    .. dropdown:: References

        .. [1] Esuli, A., Moreo, A., & Sebastiani, F. (2018).
               A Recurrent Neural Network for Sentiment Quantification.
               *CIKM*, pp. 1775–1778.
    """

    _parameter_constraints = {
        "fit_estimator": ["boolean"],
        "sample_size": [Interval(0, None, inclusive_left=False), Options([None])],
        "n_epochs": [Interval(0, None, inclusive_left=False), Options([None])],
        "tr_iter": [Interval(0, None, inclusive_left=False), Options([None])],
        "va_iter": [Interval(0, None, inclusive_left=False), Options([None])],
        "lr": [Interval(0, None, inclusive_left=False), Options([None])],
        "lstm_hidden_size": [Interval(0, None, inclusive_left=False), Options([None])],
        "lstm_nlayers": [Interval(0, None, inclusive_left=False), Options([None])],
        "bidirectional": ["boolean"],
        "qdrop_p": [Interval(0, None, inclusive_left=False), Options([None])],
        "patience": [Interval(0, None, inclusive_left=False), Options([None])],
        "checkpointdir": ["string"],
        "checkpointname": ["string"],
    }


    def __init__(
        self,
        estimator,
        embedder=None,
        fit_estimator: bool = True,
        sample_size: int = 100,
        n_epochs: int = 100,
        tr_iter: int = 500,
        va_iter: int = 100,
        lr: float = 1e-3,
        lstm_hidden_size: int = 64,
        lstm_nlayers: int = 1,
        ff_layers: Sequence[int] = (1024, 512),
        bidirectional: bool = True,
        random_state: int = None,
        qdrop_p: float = 0.5,
        patience: int = 10,
        checkpointdir: str = "./checkpoint_quanet",
        checkpointname: str | None = None,
        device: str = "cuda",
    ) -> None:

        if torch is None:
            raise ImportError("PyTorch is required to use QuaNet.")

        # save hyperparameters as attributes
        self.estimator = estimator
        self.embedder = embedder
        self.fit_estimator = fit_estimator
        self.sample_size = sample_size
        self.n_epochs = n_epochs
        self.tr_iter = tr_iter
        self.va_iter = va_iter
        self.lr = lr
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_nlayers = lstm_nlayers
        self.ff_layers = ff_layers
        self.bidirectional = bidirectional
        self.random_state = random_state
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

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y):
        """Fit QuaNet to the training data.

        Optionally fits the base estimator, then trains the LSTM network
        on artificially generated bags sampled with the UPP protocol.
        Uses early stopping based on the validation loss.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix. Must be compatible with both
            ``estimator.fit`` and ``estimator.transform``.
        y : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : QuaNet
            The fitted quantifier.

        Notes
        -----
        When ``fit_estimator=True`` the data is internally split into a
        classifier-training set (60 %), a network-training set (32 %), and
        a validation set (8 %). When ``fit_estimator=False`` only the
        train/validation split (80 %/20 %) is performed.
        """
        y = validate_data(self, y=y)
        self.classes_ = check_classes_attribute(self, np.unique(y))

        # resolve embedding source
        _embedder = self.embedder if self.embedder is not None else self.estimator
        if not hasattr(_embedder, "transform"):
            raise ValueError(
                "The estimator must have a transform() method, or pass a separate "
                "`embedder` (e.g. TfidfVectorizer, PCA, TorchClassifierWrapper) "
                "that does."
            )

        self.embedder_ = _embedder

        os.makedirs(self.checkpointdir, exist_ok=True)

        if self.fit_estimator:
            X_clf, X_rest, y_clf, y_rest = train_test_split(
                X, y, test_size=0.4, random_state=self.random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_rest, y_rest, test_size=0.2, random_state=self.random_state, stratify=y_rest
            )

            self.estimator.fit(X_clf, y_clf)
            if self.embedder is not None and hasattr(self.embedder, "fit"):
                self.embedder.fit(X_clf, y_clf)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.40, random_state=self.random_state, stratify=y
            )

        self.tr_prev = get_prev_from_labels(y, format="array")

        X_train_embeddings = _embedder.transform(X_train)
        X_val_embeddings = _embedder.transform(X_val)
        
        valid_posteriors = self.estimator.predict_proba(X_val)
        train_posteriors = self.estimator.predict_proba(X_train)

        self.val_posteriors = valid_posteriors
        self.y_val = y_val

        self.quantifiers = {
            "cc": CC(self.estimator),
            "acc": GACC(self.estimator),
            "pcc": PCC(self.estimator),
            "pacc": GPACC(self.estimator),
            "emq": EMQ(self.estimator),
        }

        self.status = {
            "tr-loss": -1.0,
            "va-loss": -1.0,
            "tr-mae": -1.0,
            "va-mae": -1.0,
        }

        numQtf = len(self.quantifiers)
        numClasses = len(self.classes_)

        self.quanet = QuaNetModule(
            doc_embedding_size=X_train_embeddings.shape[1],
            n_classes=numClasses,
            stats_size=numQtf*numClasses,
            order_by=0 if numClasses == 2 else None,
            **self.quanet_params
        ).to(self.device)
        print(self.quanet)

        self.optim = torch.optim.Adam(self.quanet.parameters(), lr=self.lr)
        early_stop = EarlyStop(
            patience=self.patience,
            lower_is_better=True,
        )

        checkpoint = self.checkpoint

        for epoch in range(self.n_epochs):
            # **CORREÇÃO: Passar embeddings em vez de X original**
            self._epoch(
                X_train_embeddings, y_train, train_posteriors, 
                self.tr_iter, epoch, early_stop, train=True
            )
            self._epoch(
                X_val_embeddings, y_val, valid_posteriors, 
                self.va_iter, epoch, early_stop, train=False
            )

            early_stop(self.status["va-loss"], epoch)
            if early_stop.IMPROVED:
                torch.save(self.quanet.state_dict(), checkpoint)
            elif early_stop.STOP:
                print(f'Training ended at epoch {early_stop.best_epoch}, loading best model parameters in {checkpoint}')
                self.quanet.load_state_dict(torch.load(checkpoint))
                break

        return self

    def _aggregate_qtf(self, posteriors, train_posteriors, y_train):
        qtf_estims = []

        for name, qtf in self.quantifiers.items():

            requirements = get_aggregation_requirements(qtf)

            if requirements.requires_train_proba and requirements.requires_train_labels:
                prev = qtf.aggregate(posteriors, train_posteriors, y_train)
            elif requirements.requires_train_labels:
                prev = qtf.aggregate(posteriors, y_train)
            else:
                # CC/PCC-style quantifiers self-source their classes; pass them
                # explicitly so absent classes still appear in the estimate.
                prev = qtf.aggregate(posteriors, classes=np.unique(y_train))

            # aggregate() may return either a dict (keyed by class) or an
            # ndarray depending on the global prevalence_return_type config.
            if isinstance(prev, dict):
                prev = np.asarray(list(prev.values()))
            qtf_estims.extend(np.asarray(prev, dtype=float).ravel())

        return qtf_estims

    
    def predict(self, X):
        """Predict class prevalences for a test bag.

        Computes posterior probabilities and document embeddings with the base
        estimator, collects simple quantification statistics (CC, GACC, PCC,
        GPACC, EMQ) as auxiliary inputs, and forwards everything through the
        trained :class:`QuaNetModule`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test feature matrix.

        Returns
        -------
        prevalences : ndarray of shape (n_classes,)
            Estimated class prevalence vector for the test bag, normalised
            to sum to 1.
        """
        estimator_function = _get_estimator_function(self)
        posteriors = getattr(self.estimator, estimator_function)(X)
        _embedder = self.embedder_ if self.embedder is not None else self.estimator
        embeddings = _embedder.transform(X)

        qtf_estims = self._aggregate_qtf(posteriors, self.val_posteriors, self.y_val)
            
        self.quanet.eval()
        with torch.no_grad():
            prevalence = self.quanet.forward(embeddings, posteriors, qtf_estims)
            if self.device.type == "cuda":
                prevalence = prevalence.cpu()
            prevalence = prevalence.numpy().flatten()
        
        return prevalence
            
    
    def _epoch(self, X, y, posteriors, iterations, epoch, early_stop, train: bool) -> None:
        mse_loss = MSELoss()

        self.quanet.train(mode=train)
        losses = []
        mae_errors = []

        sampler = UPP(
            batch_size=self.sample_size,
            n_prevalences=iterations,
            random_state= None if train else self.random_state,
        )

        for idx in sampler.split(X, y):
            X_batch = X[idx]
            y_batch = y[idx]
            posteriors_batch = posteriors[idx]
            
            qtf_estims = self._aggregate_qtf(posteriors_batch, self.val_posteriors, self.y_val)

            p_true = torch.as_tensor(
                get_prev_from_labels(y_batch, format="array", classes=self.classes_), 
                dtype=torch.float, 
                device=self.device
            ).unsqueeze(0)

            if train:
                self.optim.zero_grad()
                p_pred = self.quanet.forward(
                    X_batch, 
                    posteriors_batch, 
                    qtf_estims
                )
                loss = mse_loss(p_pred, p_true)
                mae = mae_loss(p_pred, p_true)
                loss.backward()
                self.optim.step()
            else:
                with torch.no_grad():
                    p_pred = self.quanet.forward(
                        X_batch, 
                        posteriors_batch, 
                        qtf_estims
                    )
                    loss = mse_loss(p_pred, p_true)
                    mae = mae_loss(p_pred, p_true)

            losses.append(loss.item())
            mae_errors.append(mae.item())

            mae = np.mean(mae_errors)
            mse = np.mean(losses)

            if train:
                self.status["tr-mae"] = mae
                self.status["tr-loss"] = mse
            else:
                self.status["va-mae"] = mae
                self.status["va-loss"] = mse
            

    def _check_params_colision(self, quanet_params, estimator_params):
        quanet_keys = set(quanet_params.keys())
        estimator_keys = set(estimator_params.keys())

        colision_keys = quanet_keys.intersection(estimator_keys)

        if colision_keys:
            raise ValueError(f"Parameters {colision_keys} are present in both quanet_params and estimator_params")

    def clean_checkpoint(self):
        """Remove the checkpoint file saved during training, if it exists."""
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def clean_checkpoint_dir(self):
        """Remove the entire checkpoint directory and all its contents."""
        import shutil
        shutil.rmtree(self.checkpointdir, ignore_errors=True)


def mae_loss(y_true, y_pred):
    """Compute mean absolute error between two tensors.

    Parameters
    ----------
    y_true : torch.Tensor
        Ground-truth prevalence vector(s).
    y_pred : torch.Tensor
        Predicted prevalence vector(s).

    Returns
    -------
    loss : torch.Tensor
        Scalar mean absolute error.
    """
    return torch.mean(torch.abs(y_true - y_pred))


class _SymmetricNeuralQ(BaseQuantifier):
    """Internal base for bag-labeled (symmetric) neural quantifiers.

    Subclasses must implement:

    - ``_build_network(n_features, n_classes) -> nn.Module``
        Returns the full network (FEM + BRM + QM).
    - ``_forward_batch(network, X_tensor) -> (torch.Tensor of shape (n_bags, n_classes), extras)``
        One forward pass for a batch of bags (tensor of shape
        ``(n_bags, bag_size, n_features)``); returns one prevalence vector per
        bag. ``extras`` carries any auxiliary tensors needed by the loss
        (e.g. the latent projections used by GMNet's CKA regulariser) or
        ``None``.
    """

    def fit(self, X, y):
        """Fit on individually labeled data.

        Internally generates bags by sampling from ``(X, y)``, applies Bag
        Mixer augmentation each epoch, and optimises the chosen loss directly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        y = validate_data(self, y=y)
        self.classes_ = check_classes_attribute(self, np.unique(y))
        n_classes = len(self.classes_)
        X = np.asarray(X, dtype=float)
        # Standardise features at the *data* level (fixed mean/std over the whole
        # training set). This keeps the sigmoid before the bag representation in
        # its responsive range without normalising away each bag's class
        # composition — the very signal we quantify (per-bag normalisation such
        # as BatchNorm would erase it and collapse the network to a constant).
        self._x_mean = X.mean(axis=0, keepdims=True)
        self._x_std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - self._x_mean) / self._x_std
        rng = np.random.default_rng(self.random_state)

        # Split into train / validation (80/20) and store the per-example pools;
        # ``_sample_train_bags`` draws fresh APP bags from them every epoch.
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        self._X_tr, self._y_tr = X_tr, y_tr
        self._val_bags = self._sample_bags(X_va, y_va, self.val_bags, self.bag_size, rng)

        return self._fit_loop(X_tr.shape[1], n_classes, rng)

    def _sample_bag_chunk(self, rng, k):
        """Sample ``k`` training bags — one mini-batch's worth.

        Called lazily by :meth:`_train_batches`, so only ``batch_size`` bags are
        ever materialised at once (sampling a whole epoch of large bags up front
        would exhaust RAM). Default (individual-label) behaviour: draw ``k`` fresh
        APP bags from the stored ``(X, y)`` pools and apply Bag Mixer
        augmentation within the chunk. The bag-labelled training path (see
        :class:`~mlquantify.neural.PrevalenceBagMixin`) overrides this to sample
        from a pool of real, prevalence-labelled bags.
        """
        raw = self._sample_bags(self._X_tr, self._y_tr, k, self.bag_size, rng)
        Xc = [b[0] for b in raw]
        pc = [b[1] for b in raw]
        return bag_mixer(Xc, None, pc, ratio=self.bag_mixer_ratio, rng=rng)

    def _train_batches(self, rng):
        """Yield stacked ``(X, prev)`` mini-batches for one training epoch.

        Each item is ``(np.ndarray (k, bag_size, n_features), np.ndarray
        (k, n_classes))``. Sampling *and* stacking happen here so they can be
        overlapped with GPU compute by :class:`BackgroundPrefetch`.
        """
        bs = max(int(getattr(self, "batch_size", 1)), 1)
        remaining = self.n_bags
        while remaining > 0:
            k = min(bs, remaining)
            Xc, pc = self._sample_bag_chunk(rng, k)
            yield np.stack(Xc), np.stack(pc)
            remaining -= k

    def _val_batches(self):
        """Yield stacked ``(X, prev)`` mini-batches over the fixed val bags."""
        bs = max(int(getattr(self, "batch_size", 1)), 1)
        vx = [b[0] for b in self._val_bags]
        vp = [b[1] for b in self._val_bags]
        for s in range(0, len(vx), bs):
            yield np.stack(vx[s:s + bs]), np.stack(vp[s:s + bs])

    def _fit_loop(self, n_features, n_classes, rng):
        """Build the network and run the shared training loop.

        Reads training bags each epoch from :meth:`_sample_train_bags` and the
        fixed validation bags from ``self._val_bags`` — both fit paths (the
        default ``(X, y)`` and the bag-labelled mixin) funnel through here.
        """
        self.network_ = self._build_network(n_features, n_classes).to(
            torch.device(self.device)
        )

        optim_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[
            str(getattr(self, "optimizer", "adam")).lower()
        ]
        optimiser = optim_cls(
            self.network_.parameters(),
            lr=self.lr,
            weight_decay=getattr(self, "weight_decay", 0.0),
        )
        # When ``end_lr`` is set we follow the authors' schedule: reduce the LR on
        # a validation-loss plateau and stop once it falls below ``end_lr``.
        # Otherwise we keep the simple fixed-LR + EarlyStop behaviour.
        end_lr = getattr(self, "end_lr", None)
        if end_lr is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimiser, factor=self.lr_factor, patience=self.patience, cooldown=0
            )
            early_stop = None
        else:
            scheduler = None
            early_stop = EarlyStop(patience=self.patience, lower_is_better=True)

        best_val = np.inf
        self._best_state = {k: v.clone() for k, v in self.network_.state_dict().items()}
        start_epoch = 0

        # Resume from a checkpoint if one exists (same ``checkpoint_path``).
        ckpt_path = getattr(self, "checkpoint_path", None)
        if ckpt_path and os.path.exists(ckpt_path):
            ck = torch.load(ckpt_path, map_location=torch.device(self.device), weights_only=False)
            self.network_.load_state_dict(ck["network"])
            optimiser.load_state_dict(ck["optimizer"])
            if scheduler is not None and ck.get("scheduler") is not None:
                scheduler.load_state_dict(ck["scheduler"])
            best_val = ck["best_val"]
            self._best_state = ck["best_state"]
            rng.bit_generator.state = ck["rng_state"]
            start_epoch = ck["epoch"] + 1
            if start_epoch >= self.n_epochs or ck.get("done", False):
                self.network_.load_state_dict(self._best_state)
                return self
            print(f"[resumed from {ckpt_path} at epoch {start_epoch}]", flush=True)

        # Optional tqdm progress bar (one row, updated per epoch).
        pbar = None
        if getattr(self, "verbose", False):
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=self.n_epochs, initial=start_epoch,
                            desc="training", unit="epoch")
            except ImportError:
                pbar = None

        def _save_checkpoint(epoch, done=False):
            if not ckpt_path:
                return
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            tmp = ckpt_path + ".tmp"
            torch.save({
                "epoch": epoch,
                "network": self.network_.state_dict(),
                "optimizer": optimiser.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_val": best_val,
                "best_state": self._best_state,
                "rng_state": rng.bit_generator.state,
                "done": done,
            }, tmp)
            os.replace(tmp, ckpt_path)  # atomic: never leaves a half-written file

        every = int(getattr(self, "checkpoint_every", 0) or 0)
        stopped = False
        epoch = start_epoch - 1
        for epoch in range(start_epoch, self.n_epochs):
            # Prefetch bag sampling in a worker thread so it overlaps GPU compute
            # (otherwise the GPU starves waiting on NumPy bag generation).
            tr_loss = self._run_epoch(
                BackgroundPrefetch(self._train_batches(rng)), optimiser, train=True
            )
            va_loss = self._run_epoch(
                BackgroundPrefetch(self._val_batches()), optimiser, train=False
            )

            if va_loss < best_val:
                best_val = va_loss
                self._best_state = {
                    k: v.clone() for k, v in self.network_.state_dict().items()
                }

            lr = optimiser.param_groups[0]["lr"]
            if pbar is not None:
                pbar.set_postfix(tr=f"{tr_loss:.4f}", va=f"{va_loss:.4f}", lr=f"{lr:.1e}")
                pbar.update(1)

            if scheduler is not None:
                scheduler.step(va_loss)
                if optimiser.param_groups[0]["lr"] < end_lr:
                    stopped = True
            else:
                early_stop(va_loss, epoch)
                stopped = early_stop.STOP

            if every and ((epoch + 1) % every == 0 or stopped):
                _save_checkpoint(epoch, done=stopped)
            if stopped:
                break

        # Always leave a final checkpoint marking completion.
        _save_checkpoint(epoch, done=True)

        if pbar is not None:
            pbar.close()
        self.network_.load_state_dict(self._best_state)
        return self

    def _sample_bags(self, X, y, n_bags, bag_size, rng):
        """Generate bags spanning the prevalence simplex (APP-style sampling).

        Each bag is built for a target prevalence drawn uniformly from the
        simplex: the per-class example counts are set to match that prevalence
        and the examples are sampled (with replacement) from the corresponding
        class pool. This is what gives the symmetric quantifier bags with
        *varying* prevalences to learn from — naive uniform resampling would
        produce only bags at the natural training prevalence.
        """
        n_classes = len(self.classes_)
        class_indices = [np.flatnonzero(y == c) for c in self.classes_]
        # Drop classes with no examples to avoid empty sampling pools.
        present = [k for k in range(n_classes) if len(class_indices[k]) > 0]

        bags = []
        for _ in range(n_bags):
            # Uniform draw over the simplex of the present classes.
            prev = np.zeros(n_classes, dtype=float)
            draw = rng.dirichlet(np.ones(len(present)))
            prev[present] = draw

            counts = np.floor(prev * bag_size).astype(int)
            # Distribute the rounding remainder so counts sum to bag_size.
            remainder = bag_size - counts.sum()
            if remainder > 0:
                order = np.argsort(-(prev * bag_size - counts))
                for k in order[:remainder]:
                    counts[k] += 1

            parts = []
            for k in range(n_classes):
                if counts[k] == 0:
                    continue
                pool = class_indices[k]
                chosen = pool[rng.integers(0, len(pool), size=counts[k])]
                parts.append(chosen)
            idx = np.concatenate(parts)
            rng.shuffle(idx)
            X_bag = X[idx]
            # Use the realised prevalence (counts / bag_size) as the target.
            prev_realised = counts / counts.sum()
            bags.append((X_bag, prev_realised))
        return bags

    def _run_epoch(self, batches, optimiser, train: bool):
        """Run one epoch over an iterator of stacked bag mini-batches.

        ``batches`` yields ``(X, prev)`` NumPy arrays already stacked to
        ``(n_bags, bag_size, n_features)`` / ``(n_bags, n_classes)`` — one
        optimiser step covers a whole mini-batch of bags. The iterator is
        consumed lazily (only one chunk resident) and, when wrapped with
        :class:`BackgroundPrefetch`, sampling overlaps GPU compute. Returns the
        mean loss.
        """
        device = torch.device(self.device)
        pin = device.type == "cuda"
        self.network_.train(mode=train)
        accum = max(int(getattr(self, "gradient_accumulation", 1) or 1), 1)
        losses = []
        ctx = torch.enable_grad if train else torch.no_grad
        with ctx():
            if train:
                optimiser.zero_grad()
            step = 0
            for X_np, prev_np in batches:
                X_t = torch.from_numpy(np.ascontiguousarray(X_np, dtype=np.float32))
                p_true = torch.from_numpy(np.ascontiguousarray(prev_np, dtype=np.float32))
                if pin:
                    X_t = X_t.pin_memory().to(device, non_blocking=True)
                    p_true = p_true.pin_memory().to(device, non_blocking=True)
                else:
                    X_t, p_true = X_t.to(device), p_true.to(device)
                p_pred, extras = self._forward_batch(self.network_, X_t)
                loss = self._compute_loss(p_pred, p_true, extras)
                losses.append(loss.item())
                if train:
                    # Average the gradient over ``accum`` mini-batches before
                    # stepping, so the effective batch is accum * batch_size bags.
                    (loss / accum).backward()
                    step += 1
                    if step % accum == 0:
                        optimiser.step()
                        optimiser.zero_grad()
            if train and step % accum != 0:
                optimiser.step()
                optimiser.zero_grad()
        return float(np.mean(losses))

    def _compute_loss(self, p_pred, p_true, extras=None):
        if self.loss == "rae":
            # Smooth with the actual bag size (eps = 1 / (2 * bag_size)), matching
            # the LeQua RAE definition used by the authors.
            base = rae_loss(p_pred, p_true, epsilon=1.0 / (2 * self.bag_size))
        elif self.loss == "ae":
            base = torch.mean(torch.abs(p_pred - p_true))
        else:
            base = torch.mean((p_pred - p_true) ** 2)

        # CKA regularization (only for GMNet, extras carries Z_list)
        if extras is not None and getattr(self, "cka_lambda", 0) > 0:
            base = base + self.cka_lambda * cka_regularization(extras)
        return base

    def predict(self, X):
        """Predict class prevalences for a test bag.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        prevalences : ndarray of shape (n_classes,)
        """
        device = torch.device(self.device)
        X = (np.asarray(X, dtype=float) - self._x_mean) / self._x_std
        # Add a singleton bag axis -> (1, n_samples, n_features).
        X_t = torch.as_tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
        self.network_.eval()
        with torch.no_grad():
            p_pred, _ = self._forward_batch(self.network_, X_t)
        return p_pred.cpu().numpy().flatten()


class HistNetQ(_SymmetricNeuralQ):
    r"""Neural quantifier with differentiable histogram bag representation.

    Implements the HistNetQ architecture from Pérez-Mon et al. (2024).
    Trains end-to-end on bags labeled by prevalence (symmetric approach),
    directly minimising a quantification loss (RAE by default).

    Architecture::

        FEM (feature_extractor) → DifferentiableHistogramRepresentation → QM (MLP + softmax)

    The FEM maps each example in the bag to a latent vector. The histogram
    layer summarises the bag permutation-invariantly. The QM predicts prevalences.

    Parameters
    ----------
    feature_extractor : nn.Module
        Feature Extraction Module. Maps (n_examples, n_input_features) →
        (n_examples, n_latent_features) and outputs the **raw** latent vector
        (no final ``nn.Sigmoid()``): the network squashes it into the ``[0, 1]``
        histogram range internally with a sigmoid. Input features are
        standardised at the data level inside :meth:`fit`, so the sigmoid stays
        responsive; a deep FEM with dropout (e.g. ``0.5``) works well.
    n_latent_features : int
        Output dimensionality of ``feature_extractor``. Must be specified
        explicitly so the histogram layer can be sized correctly.
    n_bins : int, default=32
        Number of histogram bins per latent feature.
    ff_layers : tuple of int, default=(512, 256)
        Hidden layer sizes in the Quantification Module (QM).
    lr : float, default=1e-3
        Initial learning rate.
    optimizer : {'adam', 'adamw'}, default='adam'
        Optimiser used for training.
    weight_decay : float, default=0.0
        L2 / decoupled weight-decay regularisation passed to the optimiser.
    end_lr : float or None, default=None
        If set, the learning rate is reduced on a validation-loss plateau
        (:class:`~torch.optim.lr_scheduler.ReduceLROnPlateau`, factor
        ``lr_factor``, patience ``patience``) and training stops once it falls
        below ``end_lr``. If ``None``, a fixed learning rate with simple
        early stopping (``patience``) is used instead.
    lr_factor : float, default=0.1
        Multiplicative LR-reduction factor used when ``end_lr`` is set.
    n_epochs : int, default=100
        Maximum training epochs.
    patience : int, default=20
        Epochs without validation improvement before the LR is reduced
        (``end_lr`` set) or training is early-stopped (``end_lr=None``).
    bag_size : int, default=500
        Number of examples per training bag.
    n_bags : int, default=1000
        Training bags generated per epoch.
    val_bags : int, default=300
        Validation bags per epoch.
    batch_size : int, default=10
        Number of bags stacked into a single mini-batch per optimiser step.
        Larger values give less noisy gradients (at higher memory cost).
    gradient_accumulation : int, default=1
        Number of mini-batches to accumulate gradients over before each
        optimiser step (effective batch size ``gradient_accumulation *
        batch_size`` bags).
    bag_mixer_ratio : float, default=0.5
        Fraction of training bags replaced by Bag Mixer augmented bags.
    loss : {'rae', 'ae', 'mse'}, default='rae'
        Loss function optimised during training.
    device : str, default='cpu'
    random_state : int or None, default=None

    References
    ----------
    Pérez-Mon, O., Moreo, A., del Coz, J.J., & González, P. (2024).
    Quantification using permutation-invariant networks based on histograms.
    *Neural Computing and Applications*, 37, 3505–3520.
    """

    def __init__(
        self,
        feature_extractor,
        n_latent_features: int,
        n_bins: int = 32,
        ff_layers=(512, 256),
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        end_lr: float = None,
        lr_factor: float = 0.1,
        n_epochs: int = 100,
        patience: int = 20,
        bag_size: int = 500,
        n_bags: int = 1000,
        val_bags: int = 300,
        batch_size: int = 10,
        gradient_accumulation: int = 1,
        bag_mixer_ratio: float = 0.5,
        loss: str = "rae",
        device: str = "cpu",
        random_state: int = None,
        verbose: bool = False,
        checkpoint_path: str = None,
        checkpoint_every: int = 0,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to use HistNetQ.")
        self.feature_extractor = feature_extractor
        self.n_latent_features = n_latent_features
        self.n_bins = n_bins
        self.ff_layers = ff_layers
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.end_lr = end_lr
        self.lr_factor = lr_factor
        self.n_epochs = n_epochs
        self.patience = patience
        self.bag_size = bag_size
        self.n_bags = n_bags
        self.val_bags = val_bags
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.bag_mixer_ratio = bag_mixer_ratio
        self.loss = loss
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every

    def _build_network(self, n_input_features, n_classes):
        from mlquantify.representations import DifferentiableHistogramRepresentation

        brm = DifferentiableHistogramRepresentation(
            n_features=self.n_latent_features, n_bins=self.n_bins
        )
        qm_input_size = brm.output_size
        layers = []
        in_size = qm_input_size
        for out_size in self.ff_layers:
            layers += [nn.Linear(in_size, out_size), nn.LeakyReLU(), nn.Dropout(0.5)]
            in_size = out_size
        layers.append(nn.Linear(in_size, n_classes))
        # Squash the FEM output into [0, 1] before binning (the histogram bins
        # live in [0, 1]). Inputs are standardised at the data level in ``fit``,
        # which keeps this sigmoid in its responsive range, so a plain sigmoid is
        # enough — no per-bag normalisation (which would erase the prevalence
        # signal).
        return nn.ModuleDict({
            "fem": self.feature_extractor,
            "to_unit": nn.Sigmoid(),
            "brm": brm,
            "qm": nn.Sequential(*layers),
        })

    def _forward_batch(self, network, X):
        # X: (B, n, n_input_features). Linear layers act on the last axis, so the
        # bag and example axes pass through untouched.
        latent = network["fem"](X)                  # (B, n, n_latent_features)
        latent = network["to_unit"](latent)         # (B, n, n_latent_features) in [0, 1]
        hist = network["brm"](latent)               # (B, n_bins * n_latent_features)
        logits = network["qm"](hist)                # (B, n_classes)
        p_pred = torch.softmax(logits, dim=-1)
        return p_pred, None                          # no extras for HistNetQ


class GMNet(_SymmetricNeuralQ):
    r"""Neural quantifier with Gaussian latent-space bag representation.

    Implements the GMNet architecture from Pérez-Mon et al. (2025).
    Uses K learnable multivariate Gaussian distributions across L independent
    latent spaces to represent bags permutation-invariantly.

    Architecture::

        L × FEM (replicated/independent projections, sigmoid output) →
        GaussianRepresentation → QM (MLP + softmax)

    Parameters
    ----------
    feature_extractor : nn.Module
        Shared base FEM. Replicated L times internally (one projection
        layer per latent space is appended, each outputting ``latent_dim``
        units with sigmoid activation).
    n_input_features : int
        Input dimensionality fed into ``feature_extractor``.
    latent_dim : int, default=8
        Dimensionality d of each Gaussian latent space. The covariance is a full
        ``d × d`` matrix per Gaussian, so keep this small (the paper uses ~3–8).
    n_gaussians : int, default=100
        Number K of Gaussians per latent space.
    n_latent : int, default=9
        Number L of independent latent spaces.
    ff_layers : tuple of int, default=(512, 256)
        Hidden layer sizes in the Quantification Module.
    cka_lambda : float, default=0.01
        Weight for CKA diversity regularization loss.
        Set to 0.0 to disable.
    lr : float, default=1e-3
        Initial learning rate.
    optimizer : {'adam', 'adamw'}, default='adam'
        Optimiser used for training.
    weight_decay : float, default=0.0
        L2 / decoupled weight-decay regularisation passed to the optimiser.
    end_lr : float or None, default=None
        If set, reduce the LR on a validation-loss plateau (factor ``lr_factor``,
        patience ``patience``) and stop once it falls below ``end_lr``; otherwise
        use a fixed LR with simple early stopping.
    lr_factor : float, default=0.1
        Multiplicative LR-reduction factor used when ``end_lr`` is set.
    n_epochs : int, default=100
    patience : int, default=20
    bag_size : int, default=1000
    n_bags : int, default=1000
    val_bags : int, default=300
    batch_size : int, default=10
        Number of bags stacked into a single mini-batch per optimiser step.
    gradient_accumulation : int, default=1
        Mini-batches to accumulate gradients over before each optimiser step.
    bag_mixer_ratio : float, default=0.5
    loss : {'rae', 'ae', 'mse'}, default='rae'
    device : str, default='cpu'
    random_state : int or None, default=None

    References
    ----------
    Pérez-Mon, O., del Coz, J.J., & González, P. (2025).
    Quantification Via Gaussian Latent Space Representations.
    arXiv:2501.13638.
    """

    def __init__(
        self,
        feature_extractor,
        n_input_features: int,
        latent_dim: int = 8,
        n_gaussians: int = 100,
        n_latent: int = 9,
        ff_layers=(512, 256),
        cka_lambda: float = 0.01,
        lr: float = 1e-3,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        end_lr: float = None,
        lr_factor: float = 0.1,
        n_epochs: int = 100,
        patience: int = 20,
        bag_size: int = 1000,
        n_bags: int = 1000,
        val_bags: int = 300,
        batch_size: int = 10,
        gradient_accumulation: int = 1,
        bag_mixer_ratio: float = 0.5,
        loss: str = "rae",
        device: str = "cpu",
        random_state: int = None,
        verbose: bool = False,
        checkpoint_path: str = None,
        checkpoint_every: int = 0,
    ):
        if torch is None:
            raise ImportError("PyTorch is required to use GMNet.")
        self.feature_extractor = feature_extractor
        self.n_input_features = n_input_features
        self.latent_dim = latent_dim
        self.n_gaussians = n_gaussians
        self.n_latent = n_latent
        self.ff_layers = ff_layers
        self.cka_lambda = cka_lambda
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.end_lr = end_lr
        self.lr_factor = lr_factor
        self.n_epochs = n_epochs
        self.patience = patience
        self.bag_size = bag_size
        self.n_bags = n_bags
        self.val_bags = val_bags
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.bag_mixer_ratio = bag_mixer_ratio
        self.loss = loss
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.checkpoint_every = checkpoint_every

    def _build_network(self, n_input_features, n_classes):
        from mlquantify.representations import GaussianRepresentation

        # L independent projection heads on top of shared FEM. Inputs are
        # standardised at the data level in ``fit``, keeping the sigmoid in its
        # responsive range (no per-bag normalisation, which would erase the
        # prevalence signal carried by each bag's composition).
        projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._get_fem_output_size(), self.latent_dim),
                nn.Sigmoid(),
            )
            for _ in range(self.n_latent)
        ])
        brm = GaussianRepresentation(
            latent_dim=self.latent_dim,
            n_gaussians=self.n_gaussians,
            n_latent=self.n_latent,
        )
        qm_input_size = brm.output_size
        layers = []
        in_size = qm_input_size
        for out_size in self.ff_layers:
            layers += [nn.Linear(in_size, out_size), nn.LeakyReLU(), nn.Dropout(0.5)]
            in_size = out_size
        layers.append(nn.Linear(in_size, n_classes))
        return nn.ModuleDict({
            "fem": self.feature_extractor,
            "projections": projections,
            "brm": brm,
            "qm": nn.Sequential(*layers),
        })

    def _get_fem_output_size(self):
        # Infer FEM output size by running a dummy forward pass
        was_training = self.feature_extractor.training
        self.feature_extractor.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, self.n_input_features)
            out = self.feature_extractor(dummy)
        self.feature_extractor.train(was_training)
        return out.shape[-1]

    def _forward_batch(self, network, X):
        # X: (B, n, n_input_features). The FEM and projections are Linear-based,
        # so they act on the last axis and keep the bag/example axes.
        fem_out = network["fem"](X)        # (B, n, fem_output_size)
        Z_list = [proj(fem_out) for proj in network["projections"]]  # each (B, n, latent_dim)
        # Stack the L latent spaces: (B, L, n, latent_dim)
        Z_stacked = torch.stack(Z_list, dim=1)
        repr_vec = network["brm"](Z_stacked)          # (B, L * n_gaussians)
        logits = network["qm"](repr_vec)              # (B, n_classes)
        p_pred = torch.softmax(logits, dim=-1)
        # CKA regulariser works on (n_examples, latent_dim) per space; flatten the
        # bag and example axes together so it sees all examples in the batch.
        extras = [z.reshape(-1, z.shape[-1]) for z in Z_list]
        return p_pred, extras

