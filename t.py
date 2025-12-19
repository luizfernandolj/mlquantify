import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import torch
from sklearn.ensemble import RandomForestClassifier
from mlquantify.neural import QuaNet
from mlquantify.likelihood import EMQ
from mlquantify.adjust_counting import CC
from mlquantify.metrics import MAE, NRAE
from mlquantify.utils import get_prev_from_labels

class PipelineWithTransform:
    """
    Wrapper to expose `transform(X)` as the embedding from a preprocessing
    pipeline, while delegating fit/predict_proba to the full pipeline.
    """
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def transform(self, X):
        # apply all steps except the final classifier as embedding
        Xt = X
        for name, step in self.pipeline.steps[:-1]:
            Xt = step.transform(Xt)
        return Xt

# Load data
X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Base classifier pipeline:
# - StandardScaler: normaliza features
# - PCA: gera embeddings de baixa dimensão
# - MLPClassifier: classificador com predict_proba
base_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10, random_state=42)),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(50, 25),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        max_iter=500,
        random_state=42,
    )),
])

# Wrapper que fornece fit, predict_proba e transform (embeddings do PCA)
learner = PipelineWithTransform(base_pipe)

# QuaNet quantifier
qmodel = QuaNet(
    learner=learner,
    fit_learner=True,
    sample_size=100,
    n_epochs=20,
    tr_iter=200,
    va_iter=50,
    lr=1e-3,
    lstm_hidden_size=64,
    lstm_nlayers=1,
    ff_layers=(128, 64),
    bidirectional=True,
    random_state=42,
    qdrop_p=0.5,
    patience=5,
    checkpointdir="./checkpoint_quanet",
    checkpointname=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

qmodel.fit(X_train, y_train)

pred_prev = qmodel.predict(X_test)
real_prev = get_prev_from_labels(y_test)

print("Real prevalences :", real_prev)
print("Pred prevalences:", pred_prev)

print("MAE :", MAE(real_prev, pred_prev))
print("NRAE:", NRAE(real_prev, pred_prev))
