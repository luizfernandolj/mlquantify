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
    pass

from mlquantify.base import BaseQuantifier
from mlquantify.base_aggregative import (
    AggregationMixin,
    SoftLearnerQMixin,
    get_aggregation_requirements,
    _get_learner_function
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

from mlquantify.adjust_counting import CC, AC, PCC, PAC
from mlquantify.likelihood import EMQ

EPS = 1e-12



class EarlyStop:
    """
    A class implementing the early-stopping condition typically used for training neural networks.

    >>> earlystop = EarlyStop(patience=2, lower_is_better=True)
    >>> earlystop(0.9, epoch=0)
    >>> earlystop(0.7, epoch=1)
    >>> earlystop.IMPROVED  # is True
    >>> earlystop(1.0, epoch=2)
    >>> earlystop.STOP  # is False (patience=1)
    >>> earlystop(1.0, epoch=3)
    >>> earlystop.STOP  # is True (patience=0)
    >>> earlystop.best_epoch  # is 1
    >>> earlystop.best_score  # is 0.7

    :param patience: the number of (consecutive) times that a monitored evaluation metric (typically obtaind in a
        held-out validation split) can be found to be worse than the best one obtained so far, before flagging the
        stopping condition. An instance of this class is `callable`, and is to be used as follows:
    :param lower_is_better: if True (default) the metric is to be minimized.
    :ivar best_score: keeps track of the best value seen so far
    :ivar best_epoch: keeps track of the epoch in which the best score was set
    :ivar STOP: flag (boolean) indicating the stopping condition
    :ivar IMPROVED: flag (boolean) indicating whether there was an improvement in the last call
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
        """
        Commits the new score found in epoch `epoch`. If the score improves over the best score found so far, then
        the patiente counter gets reset. If otherwise, the patience counter is decreased, and in case it reachs 0,
        the flag STOP becomes True.

        :param watch_score: the new score
        :param epoch: the current epoch
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

        # the entire set represents only one instance in quapy contexts, and so the batch_size=1
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
    tr_iter : int, default=500
        Number of APP samplings (training iterations) per epoch.
    va_iter : int, default=100
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
        learner,
        fit_learner: bool = True,
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

        assert hasattr(learner, "transform"), ...
        assert hasattr(learner, "predict_proba"), ...

        # save hyperparameters as attributes
        self.learner = learner
        self.fit_learner = fit_learner
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
        y = validate_data(self, y=y)
        self.classes_ = check_classes_attribute(self, np.unique(y))

        os.makedirs(self.checkpointdir, exist_ok=True)

        if self.fit_learner:
            X_clf, X_rest, y_clf, y_rest = train_test_split(
                X, y, test_size=0.4, random_state=self.random_state, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_rest, y_rest, test_size=0.2, random_state=self.random_state, stratify=y_rest
            )

            self.learner.fit(X_clf, y_clf)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.40, random_state=self.random_state, stratify=y
            )
        
        self.tr_prev = get_prev_from_labels(y, format="array")

        # **CORREÇÃO: Obter embeddings e suas dimensões**
        X_train_embeddings = self.learner.transform(X_train)
        X_val_embeddings = self.learner.transform(X_val)
        
        valid_posteriors = self.learner.predict_proba(X_val)
        train_posteriors = self.learner.predict_proba(X_train)

        self.val_posteriors = valid_posteriors
        self.y_val = y_val

        self.quantifiers = {
            "cc": CC(self.learner),
            "acc": AC(self.learner),
            "pcc": PCC(self.learner),
            "pacc": PAC(self.learner),
            "emq": EMQ(self.learner),
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
                prev = qtf.aggregate(posteriors)

            qtf_estims.extend(np.asarray(list(prev.values())))

        return qtf_estims

    
    def predict(self, X):
        
        learner_function = _get_learner_function(self)
        posteriors = getattr(self.learner, learner_function)(X)
        embeddings = self.learner.transform(X)

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
            

    def _check_params_colision(self, quanet_params, learner_params):
        quanet_keys = set(quanet_params.keys())
        learner_keys = set(learner_params.keys())

        colision_keys = quanet_keys.intersection(learner_keys)

        if colision_keys:
            raise ValueError(f"Parameters {colision_keys} are present in both quanet_params and learner_params")

    def clean_checkpoint(self):
        if os.path.exists(self.checkpoint):
            os.remove(self.checkpoint)

    def clean_checkpoint_dir(self):
        import shutil
        shutil.rmtree(self.checkpointdir, ignore_errors=True)


def mae_loss(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))




        
        
