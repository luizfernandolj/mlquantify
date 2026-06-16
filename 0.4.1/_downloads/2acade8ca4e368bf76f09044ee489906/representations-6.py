import numpy as np
import matplotlib.pyplot as plt
from mlquantify.representations import KernelMeanRepresentation

rng = np.random.default_rng(0)
X0 = rng.normal(-1.3, 0.8, (300, 6))
X1 = rng.normal(1.3, 0.8, (300, 6))
X = np.vstack([X0, X1])
y = np.r_[np.zeros(300), np.ones(300)].astype(int)

rep = KernelMeanRepresentation(kernel="linear").fit(X, y)
e0 = np.asarray(rep.transform(X[y == 0]), float)   # class-0 mean embedding
e1 = np.asarray(rep.transform(X[y == 1]), float)

feat = np.arange(len(e0))
fig, ax = plt.subplots(figsize=(6.2, 3))
ax.bar(feat - 0.19, e0, width=0.38, color="#4477aa", label="negative class")
ax.bar(feat + 0.19, e1, width=0.38, color="#cc6677", label="positive class")
ax.set_xticks(feat)
ax.set_xticklabels([f"f{i}" for i in feat])
ax.set_xlabel("feature")
ax.set_ylabel("mean embedding")
ax.legend(fontsize=8)
fig.tight_layout()