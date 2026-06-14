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
        "fit_estimator": [Interval(0, None, inclusive_left=False), Options([None])],
        "sample_size": [Interval(0, None, inclusive_left=False), Options([None])],
        "n_epochs": [Interval(0, None, inclusive_left=False), Options([None])],
        "tr_iter": [Interval(0, None, inclusive_left=False), Options([None])],
        "va_iter": [Interval(0, None, inclusive_left=False), Options([None])],
        "lr": [Interval(0, None, inclusive_left=False), Options([None])],
        "lstm_hidden_size": [Interval(0, None, inclusive_left=False), Options([None])],
        "lstm_nlayers": [Interval(0, None, inclusive_left=False), Options([None])],
        "bidirectional": [Interval(0, None, inclusive_left=False), Options([None])],
        "qdrop_p": [Interval(0, None, inclusive_left=False), Options([None])],
        "patience": [Interval(0, None, inclusive_left=False), Options([None])],
        "checkpointdir": ["string"],
        "checkpointname": ["string"],
    }


    def __init__(
        self,
        estimator,
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

        assert hasattr(estimator, "transform"), ...
        assert hasattr(estimator, "predict_proba"), ...

        # save hyperparameters as attributes
        self.estimator = estimator
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

        os.makedirs(self.checkpointdir, exist_ok=True)

        if self.fit_estimator:
            X_clf, X_rest, y_clf, y_rest = train_test_split(
                X, y, test_size=0.4, random_state=self.random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_rest, y_rest, test_size=0.2, random_state=self.random_state, stratify=y_rest
            )

            self.estimator.fit(X_clf, y_clf)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.40, random_state=self.random_state, stratify=y
            )
        
        self.tr_prev = get_prev_from_labels(y, format="array")

        # **CORREÇÃO: Obter embeddings e suas dimensões**
        X_train_embeddings = self.estimator.transform(X_train)
        X_val_embeddings = self.estimator.transform(X_val)
        
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

        # **CORREÇÃO: Use a dimensão dos embeddings, não das features originais**
        self.quanet = QuaNetModule(
            doc_embedding_size=X_train_embeddings.shape[1],  # ← MUDANÇA AQUI
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

            qtf_estims.extend(np.asarray(list(prev.values())))

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
        embeddings = self.estimator.transform(X)

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




        
        
