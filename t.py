from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from mlquantify.neural import QuaNet
from mlquantify.metrics import MAE, NRAE
from mlquantify.utils import get_prev_from_labels
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


class PipelineWithTransform:
    """
    Wrapper to expose `transform(X)` as the embedding from a preprocessing pipeline,
    while delegating fit/predict_proba to the full pipeline.
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
    X, y, test_size=0.3, random_state=42
)

# Preprocessing + classifier pipeline
base_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10)),   # embeddings will be 10-D
    ("clf", MLPClassifier(random_state=42))
])

# Wrap pipeline so QuaNet can use transform()
learner = PipelineWithTransform(base_pipe)

# QuaNet quantifier
qmodel = QuaNet(learner)
qmodel.fit(X_train, y_train)

pred_prev = qmodel.predict(X_test)
real_prev = get_prev_from_labels(y_test)

print("MAE :", MAE(real_prev, pred_prev))
print("NRAE:", NRAE(real_prev, pred_prev))
